import torch
import unittest

from nhsmm.constants import EPS, DTYPE
from nhsmm.distributions import Categorical


class TestCategorical(unittest.TestCase):

    def test_init_from_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=DTYPE)
        dist = Categorical(logits=logits)
        print("probs:", dist.probs)
        print("log_probs:", dist.log_probs)
        self.assertTrue(torch.allclose(dist.probs, torch.softmax(logits, dim=-1)))
        self.assertTrue(torch.allclose(dist.log_probs, torch.log_softmax(logits, dim=-1)))

    def test_init_from_probs(self):
        probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=DTYPE)
        dist = Categorical(probs=probs)
        print("probs:", dist.probs)
        self.assertTrue(torch.allclose(dist.probs, probs))

    def test_sample_shape(self):
        logits = torch.randn(4, 5)
        dist = Categorical(logits=logits)
        s = dist.sample()
        print("sample:", s)
        self.assertEqual(s.shape, torch.Size([4]))

        s2 = dist.sample(sample_shape=(10,))
        print("sample with shape (10,):", s2)
        self.assertEqual(s2.shape, torch.Size([10, 4]))

    def test_rsample_soft(self):
        logits = torch.randn(3, 4)
        dist = Categorical(logits=logits)
        r = dist.rsample()
        print("rsample soft:", r)
        self.assertEqual(r.shape, logits.shape)
        self.assertTrue(torch.allclose(r.sum(dim=-1), torch.ones(3)))

    def test_rsample_hard(self):
        logits = torch.randn(3, 4)
        dist = Categorical(logits=logits)
        r = dist.rsample(hard=True)
        print("rsample hard:", r)
        self.assertEqual(r.shape, logits.shape)
        self.assertTrue(torch.all(r.sum(dim=-1) == 1))
        self.assertTrue(torch.all((r == 0) | (r == 1)))

    def test_log_prob(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=DTYPE)
        dist = Categorical(logits=logits)
        v = torch.tensor([2])
        lp = dist.log_prob(v)
        expected = torch.log_softmax(logits, dim=-1)[0, 2]
        print("log_prob:", lp)
        self.assertAlmostEqual(lp.item(), expected.item())

    def test_log_prob_broadcast(self):
        logits = torch.randn(2, 3)
        dist = Categorical(logits=logits)
        v = torch.tensor([1, 2])
        lp = dist.log_prob(v)
        print("log_prob broadcast:", lp)
        self.assertEqual(lp.shape, torch.Size([2]))

    def test_entropy(self):
        logits = torch.randn(2, 5)
        dist = Categorical(logits=logits)
        ent = dist.entropy()
        manual = -(dist.probs * dist.log_probs).sum(dim=-1)
        print("entropy:", ent)
        self.assertTrue(torch.allclose(ent, manual))

    def test_mode(self):
        logits = torch.tensor([[1.0, 5.0, 0.5]])
        dist = Categorical(logits=logits)
        m = dist.mode()
        print("mode:", m)
        self.assertEqual(m.item(), 1)

    # --- Enhanced batch/timestep/sequence tests ---
    def test_rsample_batch_timestep(self):
        logits = torch.randn(2, 3, 4)  # [batch, timestep, categories]
        dist = Categorical(logits=logits)
        r = dist.rsample()
        print("rsample batch/timestep:", r.shape)
        self.assertEqual(r.shape, logits.shape)

        r_hard = dist.rsample(hard=True)
        print("rsample hard batch/timestep:", r_hard.shape)
        self.assertEqual(r_hard.shape, logits.shape)
        self.assertTrue(torch.all(r_hard.sum(dim=-1) == 1))

    def test_sample_sequence_expansion(self):
        logits = torch.randn(3, 4, 5)  # [batch, timestep, categories]
        dist = Categorical(logits=logits)
        s = dist.sample(sample_shape=(6,))
        print("sample sequence expansion:", s.shape)
        self.assertEqual(s.shape, torch.Size([6, 3, 4]))

    def test_multiple_sampling_consistency(self):
        logits = torch.randn(6, 5)
        dist = Categorical(logits=logits)
        s = dist.sample((20,))
        print("multiple samples:", s.shape)
        self.assertEqual(s.shape, torch.Size([20, 6]))
        self.assertTrue(((s >= 0) & (s < 5)).all())


if __name__ == '__main__':
    unittest.main()
