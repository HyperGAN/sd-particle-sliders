"""Synthetic integration tests on the pinned stock DDIM scheduler and 44-hook roster."""

from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from diffusers import DDIMScheduler

import sd_apsr_replay as replay


class FakeUNet(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float16), requires_grad=False)
        self.config = SimpleNamespace(in_channels=1, cross_attention_dim=1)
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in layers)
        self.native_modules = dict(zip(layers, self.blocks, strict=True))
        self.outputs = []
        self.skip_at = None
        self.transpose_at = None

    def forward(self, model_input, timestep, *, encoder_hidden_states):
        state = encoder_hidden_states.reshape(2, 1, 1, 1).expand_as(model_input).clone()
        spatial = torch.arange(6, dtype=state.dtype).reshape(1, 2, 3) / 32
        state[1] += spatial + (int(timestep) // 35) / 64
        if len(self.outputs) == self.transpose_at:
            state = state.transpose(-2, -1)
        if len(self.outputs) == self.skip_at:
            blocks = self.blocks[:-1]
        else:
            blocks = self.blocks
        for block in blocks:
            state = block(state)
        self.outputs.append(state)
        return SimpleNamespace(sample=state)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1)
        self.calls = 0

    def decode(self, latents):
        self.calls += 1
        return SimpleNamespace(sample=latents.repeat(1, 3, 1, 1))


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.layers, self.timesteps, self.coords = replay._stock_clock()
        self.unet = FakeUNet(self.layers).eval()
        self.vae = FakeVAE().eval()
        self.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            num_train_timesteps=1000, clip_sample=False, prediction_type="epsilon",
            set_alpha_to_one=True, steps_offset=0, timestep_spacing="leading",
        )
        self.packet = {
            "schema": replay.RAW_SCHEMA,
            "weights": "raw",
            "ema_decay": None,
            "completed_optimizer_updates": 2,
            "raw_probes": {
                name: {
                    "direction.weight": torch.full((1, 1), 0.125),
                    "probe.weight": torch.full((1, 1), 0.125),
                    "probe.bias": torch.zeros(1),
                }
                for name in self.layers
            },
        }
        self.inputs = dict(
            initial_noise=torch.zeros((1, 1, 2, 3), dtype=torch.float16),
            negative=torch.tensor([[[0.125]]], dtype=torch.float16),
            positive=torch.tensor([[[0.25]]], dtype=torch.float16),
            guidance=1.0,
        )
        self.addCleanup(patch.stopall)
        patch.object(replay, "_require_stock_models").start()
        patch.object(replay, "_native_blocks", lambda unet: unet.native_modules).start()

    def compare(self, *, alpha=0.5, updates=2, owner=lambda: True):
        return replay.render_apsr_comparison(
            self.unet, self.vae, self.scheduler, raw_packet=self.packet,
            qualified_updates=updates, alpha=alpha, owner=owner, **self.inputs,
        )

    def test_exact_stock_clock_zero_parity_and_no_rng_consumption(self):
        rng = torch.random.get_rng_state()
        out = self.compare(alpha=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual((out.timesteps, out.coordinates), (self.timesteps, self.coords))
        self.assertEqual((len(self.layers), len(out.timesteps), self.vae.calls), (44, 28, 3))
        for arm in (out.adaptive, out.constant):
            self.assertTrue(torch.equal(arm.pixels, out.off.pixels))
            self.assertTrue(all(not field.any() for field in arm.coefficients.values()))
        self.assertIsNone(out.off.coefficients)
        n = len(out.timesteps)
        self.assertTrue(all(torch.equal(self.unet.outputs[i], self.unet.outputs[(i % n)])
                            for i in range(n, len(self.unet.outputs))))
        self.assertTrue(all(audit["exact_dose_positions"] == 6 for audit in out.rounding.values()))
        self.assertTrue(all(not block._forward_hooks for block in self.unet.blocks))

    def test_adaptive_causality_and_actual_matched_constant_bits(self):
        out = self.compare()
        n = len(out.timesteps)
        states = self.unet.outputs
        self.assertEqual(len(states), 3 * n)
        self.assertEqual(tuple(out.adaptive.coefficients), self.layers)
        name = self.layers[0]
        first_layer = out.adaptive.coefficients[name]
        self.assertFalse(torch.equal(first_layer[0], first_layer[-1]))
        self.assertFalse(torch.equal(first_layer[:, 0], first_layer[:, -1]))
        for index in range(n):
            expected_adaptive = states[index][1]
            expected_constant = states[index][1]
            for layer in self.layers:
                adaptive_field = expected_adaptive.float() * 0.125 * 0.5
                reported = out.adaptive.coefficients[layer][index]
                self.assertTrue(torch.equal(adaptive_field.flatten(), reported))
                expected_adaptive = (expected_adaptive.float() + adaptive_field * 0.125).half()
                constant_field = out.constant_fields[layer]
                self.assertTrue(torch.equal(out.constant.coefficients[layer][index], constant_field.flatten()))
                expected_constant = (expected_constant.float() + constant_field * 0.125).half()
            self.assertTrue(torch.equal(states[n + index][0], states[index][0]))
            self.assertTrue(torch.equal(states[2 * n + index][0], states[index][0]))
            self.assertTrue(torch.equal(states[n + index][1], expected_adaptive))
            self.assertTrue(torch.equal(states[2 * n + index][1], expected_constant))

        widths = [abs(Fraction(a) - Fraction(b)) for a, b in zip(out.coordinates, out.coordinates[1:])]
        for column, chosen in zip(first_layer.T, out.constant_fields[name].flatten(), strict=True):
            ideal = sum(w * Fraction(float(v)) for w, v in zip(widths, column, strict=True)) / sum(widths)
            actual_error = abs(ideal - Fraction(float(chosen)))
            for direction in (-float("inf"), float("inf")):
                neighbor = torch.nextafter(chosen, torch.tensor(direction))
                self.assertLessEqual(actual_error, abs(ideal - Fraction(float(neighbor))))

    def test_raw_and_strength_failures_refuse_before_any_prediction(self):
        for strength in (None, True, -1, float("nan"), 1e90):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "strength"):
                self.compare(alpha=strength)
        with self.assertRaisesRegex(ValueError, "independently qualified RAW"):
            self.compare(updates=3)
        self.packet["ema_decay"] = 0.99
        with self.assertRaisesRegex(ValueError, "no EMA"):
            self.compare()
        self.packet["ema_decay"] = None
        self.packet["raw_probes"].pop(self.layers[-1])
        with self.assertRaisesRegex(ValueError, "every probe"):
            self.compare()
        self.assertEqual(self.unet.outputs, [])
        self.assertEqual(self.vae.calls, 0)

    def test_clock_and_installed_source_drift_refuse_before_any_prediction(self):
        with patch.object(replay, "SOURCE_PINS", ((replay.PromptSteering, "0" * 64),)):
            with self.assertRaisesRegex(ValueError, "pinned replay runtime"):
                self.compare()
        self.scheduler.register_to_config(steps_offset=1)
        with self.assertRaisesRegex(ValueError, "DDIM scheduler configuration"):
            self.compare()
        self.assertEqual(self.unet.outputs, [])
        self.assertEqual(self.vae.calls, 0)

    def test_scheduler_rejects_changed_flags_and_live_alpha_or_initial_noise(self):
        for name, value in (("clip_sample", True), ("thresholding", True),
                            ("prediction_type", "v_prediction")):
            with self.subTest(name=name):
                self.scheduler.register_to_config(**{name: value})
                with self.assertRaisesRegex(ValueError, "DDIM scheduler configuration"):
                    self.compare()
                self.scheduler.register_to_config(**{name: replay.STOCK_DDIM_CONFIG[name]})
        self.scheduler.init_noise_sigma = 2.0
        with self.assertRaisesRegex(ValueError, "initial noise scale"):
            self.compare()
        self.scheduler.init_noise_sigma = 1.0
        self.scheduler.alphas_cumprod[-1] /= 2
        with self.assertRaisesRegex(ValueError, "DDIM coefficients"):
            self.compare()
        self.assertEqual(self.unet.outputs, [])
        self.assertEqual(self.vae.calls, 0)

    def test_missing_adaptive_hook_refuses_and_cleans_up(self):
        self.unet.skip_at = len(self.timesteps)
        with self.assertRaisesRegex(ValueError, "required boundaries"):
            self.compare()
        self.assertEqual(self.vae.calls, 1)
        self.assertTrue(all(not block._forward_hooks for block in self.unet.blocks))

    def test_fixed_arm_rejects_changed_spatial_lattice_and_cleans_up(self):
        self.unet.transpose_at = 2 * len(self.timesteps)
        with self.assertRaisesRegex(ValueError, "spatial lattice changed"):
            self.compare()
        self.assertEqual(self.vae.calls, 2)
        self.assertTrue(all(not block._forward_hooks for block in self.unet.blocks))

    def test_resolved_constants_keep_binary32_ties_to_even(self):
        even = torch.tensor(1.0, dtype=torch.float32)
        odd = torch.nextafter(even, torch.tensor(float("inf")))
        next_even = torch.nextafter(odd, torch.tensor(float("inf")))
        trace = torch.stack((torch.stack((even, odd)), torch.stack((odd, next_even))))
        field, audit = replay._match_field(trace, (1, 2), (2.0, 1.0, 0.0))
        self.assertTrue(torch.equal(field, torch.stack((even, next_even))[None, :]))
        self.assertEqual(audit["exact_dose_positions"], 0)
        self.assertEqual(audit["max_absolute_coefficient_dose_rounding_error"], "1/8388608")
        with self.assertRaisesRegex(ValueError, "binary32"):
            replay._match_field(trace.double(), (1, 2), (2.0, 1.0, 0.0))

    def test_owner_loss_after_one_adaptive_step_prevents_decode_and_fixed_arm(self):
        n = len(self.timesteps)
        with self.assertRaises(PermissionError):
            self.compare(owner=lambda: len(self.unet.outputs) <= n)
        self.assertEqual(len(self.unet.outputs), n + 1)
        self.assertEqual(self.vae.calls, 1)
        self.assertTrue(all(not block._forward_hooks for block in self.unet.blocks))

    def test_technical_reuses_prior_stock_witness_and_only_executes_two_new_arms(self):
        pixels, latent, _ = replay.render_ddim(
            self.unet, self.vae, self.scheduler, steps=len(self.timesteps), **self.inputs)
        stock = self.unet.outputs[:]
        reference = {
            "initial_native_latent": replay.tensor_witness(self.inputs["initial_noise"]),
            "unconditional_CLIP": replay.tensor_witness(self.inputs["negative"]),
            "positive_CLIP": replay.tensor_witness(self.inputs["positive"]),
            "last_native_latent": replay.tensor_witness(latent),
            "pixel_sha256": replay.tensor_witness(pixels)["sha256"],
        }
        self.unet.outputs.clear()
        self.vae.calls = 0
        rng = torch.random.get_rng_state()
        result = replay.verify_apsr_zero_technical(
            self.unet, self.vae, self.scheduler, widths=dict.fromkeys(self.layers, 1),
            reference=reference, owner=lambda: True, **self.inputs)
        self.assertEqual(result, dict(native_UNet_forwards=56, native_VAE_calls=2,
                                     stock_OFF_forwards=0, equal_pixel_digest=True,
                                     equal_final_latent_digest=True, zero_coefficients=True,
                                     native_boundaries=44, ema_decay=None))
        self.assertEqual(self.vae.calls, 2)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(all(torch.equal(first, second) for first, second in
                            zip(stock + stock, self.unet.outputs, strict=True)))
        self.unet.outputs.clear()
        self.vae.calls = 0
        reference["positive_CLIP"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stock input changed"):
            replay.verify_apsr_zero_technical(
                self.unet, self.vae, self.scheduler, widths=dict.fromkeys(self.layers, 1),
                reference=reference, owner=lambda: True, **self.inputs)
        self.assertEqual(self.unet.outputs, [])
        self.assertEqual(self.vae.calls, 0)

    def test_refuses_wrong_vae_precision_before_forward(self):
        self.vae.half()
        with self.assertRaisesRegex(ValueError, "FP32 stock VAE"):
            self.compare()
        self.assertEqual(self.unet.outputs, [])


if __name__ == "__main__":
    unittest.main()
