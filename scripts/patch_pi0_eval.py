"""Backport virtual-camera validation and restore checkpoint aliases for tied parameters."""
from pathlib import Path
import shutil
root = Path('/root/autodl-tmp/lerobot-src/lerobot/src/lerobot')
factory = root / 'policies/factory.py'
modeling = root / 'policies/pi0/modeling_pi0.py'
for path in (factory, modeling):
    backup = path.with_suffix(path.suffix + '.before_eval_fix')
    assert not backup.exists(), backup
    shutil.copy2(path, backup)
text = factory.read_text()
old = '        expected_features = set(cfg.input_features.keys()) | set(cfg.output_features.keys())'
new = '''        expected_features = set(cfg.input_features.keys()) | set(cfg.output_features.keys())
        # PI0 synthesizes missing virtual cameras as masked padding in _preprocess_images.
        if cfg.type == "pi0" and getattr(cfg, "empty_cameras", 0):
            virtual = {key for key in cfg.input_features if key.startswith("observation.images.empty_camera_")}
            if len(virtual) > cfg.empty_cameras:
                raise ValueError("Virtual camera count exceeds configured empty_cameras")
            expected_features -= virtual'''
assert text.count(old) == 1
factory.write_text(text.replace(old, new))
text = modeling.read_text()
old = '            # Load the remapped state dict into the model\n'
new = '''            # Safetensors stores tied parameters only once. Restore only genuine aliases.
            current_state = model.state_dict()
            for missing_key in current_state.keys() - remapped_state_dict.keys():
                parameter = current_state[missing_key]
                aliases = [key for key in remapped_state_dict if key in current_state
                           and current_state[key].data_ptr() == parameter.data_ptr()
                           and current_state[key].shape == parameter.shape]
                if aliases:
                    remapped_state_dict[missing_key] = remapped_state_dict[aliases[0]]
                    print(f"Restored tied checkpoint alias: {missing_key} <- {aliases[0]}")
            # Load the remapped state dict into the model
'''
assert text.count(old) == 1
text = text.replace(old, new)
old = '            print(f"Warning: Could not remap state dict keys: {e}")'
assert text.count(old) == 1
text = text.replace(old, '            raise RuntimeError("Failed to load pretrained PI0 weights completely") from e')
modeling.write_text(text)
print('Compatibility patches applied; original files backed up.')
