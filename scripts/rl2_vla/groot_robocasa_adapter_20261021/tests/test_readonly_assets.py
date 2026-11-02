from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from readonly_assets import stage_xml

def test_bytes_and_relative_asset_identity(tmp_path):
    (tmp_path/'visual').mkdir();asset=tmp_path/'visual/mesh.obj';asset.write_bytes(b'unchanged mesh')
    src=tmp_path/'model.xml';src.write_bytes(b'<mujoco><asset><mesh file="visual/mesh.obj"/></asset></mujoco>')
    staged=Path(stage_xml(src))
    assert staged.read_bytes()==src.read_bytes()
    assert (staged.parent/'visual/mesh.obj').resolve()==asset.resolve()
    generated=staged.parent/'new_temporary.xml';generated.write_text('temporary')
    assert not (tmp_path/'new_temporary.xml').exists()
    assert asset.read_bytes()==b'unchanged mesh'
    assert Path(stage_xml(src))==staged

def test_parent_relative_rejected(tmp_path):
    src=tmp_path/'model.xml';src.write_text('<mujoco><asset><mesh file="../mesh.obj"/></asset></mujoco>')
    with pytest.raises(RuntimeError,match='Parent-relative'):stage_xml(src)
