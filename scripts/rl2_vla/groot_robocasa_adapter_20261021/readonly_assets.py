"""Isolate RoboCasa's temporary object XML writes without changing source assets.
Copies only the XML bytes into /tmp; sibling assets are symlinks to read-only mounts.
No physics attributes are rewritten. The sandbox persists for the process lifetime.
"""
import hashlib,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
_SANDBOX=None
_FOLDERS={}

def stage_xml(path):
    global _SANDBOX
    source=Path(path).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    if _SANDBOX is None:_SANDBOX=tempfile.TemporaryDirectory(prefix='qam-robocasa-xml-')
    folder=source.parent
    if folder not in _FOLDERS:
        dest=Path(_SANDBOX.name)/hashlib.sha256(str(folder).encode()).hexdigest()[:20]
        dest.mkdir()
        for child in folder.iterdir():
            if child.suffix=='.xml' and child.is_file():
                (dest/child.name).write_bytes(child.read_bytes())
            else:(dest/child.name).symlink_to(child,target_is_directory=child.is_dir())
        _FOLDERS[folder]=dest
    dest=_FOLDERS[folder]/source.name
    if dest.read_bytes()!=source.read_bytes():raise RuntimeError('XML staging changed bytes')
    tree=ET.parse(source)
    for elem in tree.iter():
        f=elem.get('file')
        if f and not Path(f).is_absolute():
            if '..' in Path(f).parts:raise RuntimeError('Parent-relative asset requires explicit staging review')
            # Compiler subdirectories are preserved by the sibling symlinks.
    return str(dest)

def install():
    from robocasa.models.objects.objects import MJCFObject
    if getattr(MJCFObject,'_qam_readonly_stage',False):return
    original=MJCFObject.__init__
    def init(self,name,mjcf_path,*args,**kwargs):
        original(self,name,stage_xml(mjcf_path),*args,**kwargs)
        self.mjcf_path=mjcf_path
    MJCFObject.__init__=init
    MJCFObject._qam_readonly_stage=True
