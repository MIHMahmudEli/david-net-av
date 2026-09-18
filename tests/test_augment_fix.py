"""Test that augment_batch handles bad channel counts correctly."""
import torch
from src.data.augment import augment_batch, VideoAugmentor, AudioAugmentor


def test_augment_normal():
    batch = {"video": torch.randn(4, 16, 3, 224, 224), "audio": torch.randn(4, 64000)}
    v_aug = VideoAugmentor(p=1.0)
    a_aug = AudioAugmentor(p=1.0)
    result = augment_batch(batch, v_aug, a_aug)
    assert result["video"].shape == (4, 16, 3, 224, 224), f"Got {result['video'].shape}"
    assert result["audio"].shape == (4, 64000)
    print("[PASS] test_augment_normal")


def test_augment_bad_channels_186():
    batch = {"video": torch.randn(4, 16, 186, 224, 224), "audio": torch.randn(4, 64000)}
    v_aug = VideoAugmentor(p=1.0)
    a_aug = AudioAugmentor(p=1.0)
    result = augment_batch(batch, v_aug, a_aug)
    assert result["video"].shape == (4, 16, 3, 224, 224), f"Got {result['video'].shape}"
    print("[PASS] test_augment_bad_channels_186")


def test_augment_bad_channels_222():
    batch = {"video": torch.randn(4, 16, 222, 224, 224), "audio": torch.randn(4, 64000)}
    v_aug = VideoAugmentor(p=1.0)
    a_aug = AudioAugmentor(p=1.0)
    result = augment_batch(batch, v_aug, a_aug)
    assert result["video"].shape == (4, 16, 3, 224, 224), f"Got {result['video'].shape}"
    print("[PASS] test_augment_bad_channels_222")


def test_augment_no_aug():
    batch = {"video": torch.randn(4, 16, 186, 224, 224), "audio": torch.randn(4, 64000)}
    result = augment_batch(batch, None, None)
    assert result["video"].shape == (4, 16, 3, 224, 224), f"Got {result['video'].shape}"
    print("[PASS] test_augment_no_aug")


def test_augment_list_of_tensors():
    batch = {
        "video": [torch.randn(16, 3, 224, 224) for _ in range(4)],
        "audio": [torch.randn(64000) for _ in range(4)],
    }
    v_aug = VideoAugmentor(p=1.0)
    a_aug = AudioAugmentor(p=1.0)
    result = augment_batch(batch, v_aug, a_aug)
    assert result["video"].shape == (4, 16, 3, 224, 224), f"Got {result['video'].shape}"
    print("[PASS] test_augment_list_of_tensors")


def test_gaussian_blur_various_sigmas():
    """Test Gaussian blur with various kernel sizes (k=1,3,5)."""
    aug = VideoAugmentor(p=1.0)
    for _ in range(20):
        frames = torch.randn(16, 3, 224, 224)
        result = aug(frames)
        assert result.shape == (16, 3, 224, 224), f"Got {result.shape}"
    print("[PASS] test_gaussian_blur_various_sigmas")


if __name__ == "__main__":
    test_augment_normal()
    test_augment_bad_channels_186()
    test_augment_bad_channels_222()
    test_augment_no_aug()
    test_augment_list_of_tensors()
    test_gaussian_blur_various_sigmas()
    print("\nALL AUGMENT FIX TESTS PASSED")
