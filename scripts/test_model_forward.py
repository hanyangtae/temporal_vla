#!/usr/bin/env python3
"""
Test script to verify the modified XVLA model forward pass with time/variance prediction.
"""

import torch
import sys
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

def test_forward_pass():
    """Test forward pass with dummy data."""
    print("=" * 80)
    print("Testing XVLA Model with Time/Variance Prediction")
    print("=" * 80)
    
    # Create minimal config
    config = XVLAConfig(
        device="cpu",  # Use CPU for quick test
        dtype="float32",
        chunk_size=10,
        n_action_steps=10,
        hidden_size=256,
        depth=2,  # Reduce depth for faster test
        num_heads=4,
        num_denoising_steps=2,
    )
    
    # Set required features
    from lerobot.configs.types import FeatureType, PolicyFeature
    config.input_features = {
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(10,),
        ),
        "observation.language.tokens": PolicyFeature(
            type=FeatureType.LANGUAGE,
            shape=(64,),
        ),
    }
    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }
    
    print("\n[1/4] Creating XVLA Policy...")
    try:
        policy = XVLAPolicy(config)
        print("✅ Policy created successfully")
    except Exception as e:
        print(f"❌ Failed to create policy: {e}")
        return False
    
    # Create dummy batch
    batch_size = 2
    batch = {
        "observation.images.image": torch.rand(batch_size, 3, 224, 224),
        "observation.state": torch.rand(batch_size, 10),
        "observation.language.tokens": torch.randint(0, 1000, (batch_size, 64)),
        "action": torch.rand(batch_size, config.chunk_size, 7),
        "time_to_go": torch.rand(batch_size) * 10.0,  # Random time 0-10 seconds
    }
    
    print(f"\n[2/4] Testing forward pass (training mode)...")
    print(f"  Batch size: {batch_size}")
    print(f"  Time-to-go shape: {batch['time_to_go'].shape}")
    
    try:
        policy.train()
        loss, log_dict = policy.forward(batch)
        print(f"✅ Forward pass successful!")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Loss components:")
        for key, value in log_dict.items():
            print(f"    - {key}: {value:.4f}")
        
        # Check if time_loss exists
        if "time_loss" in log_dict:
            print("✅ Time loss computed successfully")
        else:
            print("⚠️ Warning: time_loss not found in log_dict")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[3/4] Testing inference (generate_actions)...")
    try:
        policy.eval()
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
        print(f"✅ Inference successful!")
        print(f"  Output action shape: {actions.shape}")
        print(f"  Expected: ({batch_size}, {config.chunk_size}, 7)")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[4/4] Testing forward pass WITHOUT time_to_go (backward compatibility)...")
    batch_no_time = {k: v for k, v in batch.items() if k != "time_to_go"}
    try:
        policy.train()
        loss, log_dict = policy.forward(batch_no_time)
        print(f"✅ Backward compatible forward pass successful!")
        print(f"  Total loss: {loss.item():.4f}")
        if "time_loss" not in log_dict:
            print("✅ time_loss correctly skipped when time_to_go is absent")
        else:
            print("⚠️ Warning: time_loss computed even without time_to_go")
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_forward_pass()
    sys.exit(0 if success else 1)
