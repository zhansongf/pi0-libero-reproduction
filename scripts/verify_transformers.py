"""Verify the installed Pi0-specific Transformers fork without loading model weights."""
import json
from pathlib import Path
import transformers
from transformers.models.siglip import check
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

result = {
    'transformers_version': transformers.__version__,
    'pi0_patch_valid': check.check_whether_transformers_replace_is_installed_correctly(),
    'pi0_policy_import': PI0Policy.__name__,
    'installed_path': transformers.__file__,
}
assert result['transformers_version'] == '4.53.3'
assert result['pi0_patch_valid'] is True
Path('/root/autodl-tmp/lerobot-env-records/transformers-verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
