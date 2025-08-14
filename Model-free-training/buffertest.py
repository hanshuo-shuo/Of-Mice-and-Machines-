import unittest
import numpy as np
from gymnasium import spaces
from ptsdbuffer import PTSDReplayBuffer

class TestPTSDReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 10
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.buffer = PTSDReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device="cpu"
        )

    def test_negative_sampling(self):
        # Add negative reward experiences
        for _ in range(50):
            self.buffer.add(
                np.random.rand(3), np.random.rand(3), np.array([0]), np.array([-1]), np.array([False]), [{}]
            )
        samples = self.buffer.sample(self.batch_size)
        # Check that most samples have negative rewards
        negative_rewards = samples.rewards < 0
        self.assertTrue(np.sum(negative_rewards) >= self.batch_size * 0.9)

    def test_mixed_sampling(self):
        # Add mixed reward experiences
        for _ in range(25):
            self.buffer.add(
                np.random.rand(3), np.random.rand(3), np.array([0]), np.array([-1]), np.array([False]), [{}]
            )
        for _ in range(25):
            self.buffer.add(
                np.random.rand(3), np.random.rand(3), np.array([0]), np.array([1]), np.array([False]), [{}]
            )
        samples = self.buffer.sample(self.batch_size)
        # Check that most samples have negative rewards
        negative_rewards = samples.rewards < 0
        self.assertTrue(np.sum(negative_rewards) >= self.batch_size * 0.9)

    def test_no_negative_rewards(self):
        # Add only positive reward experiences
        for _ in range(50):
            self.buffer.add(
                np.random.rand(3), np.random.rand(3), np.array([0]), np.array([1]), np.array([False]), [{}]
            )
        samples = self.buffer.sample(self.batch_size)
        # Check that samples are still returned
        self.assertEqual(len(samples.rewards), self.batch_size)

if __name__ == "__main__":
    unittest.main()