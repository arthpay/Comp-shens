import os
import re
from vapoursynth import core
from awsmfunc import FrameInfo
from vspreview import set_output

filenum=0
files = sorted([f for f in os.listdir('.') if f.endswith('.mkv') or f.endswith('.m2ts') or f.endswith('.mp4') or f.endswith('.webm')])
for file in files:

	sourcee = re.sub("\[.*?\]|\(.*?\}|\{.*?\}|\.+$", "", file).strip()
	clip = core.lsmas.LWLibavSource(file)
	clip = clip.std.SetFrameProp('Name', data = sourcee)
	clip= FrameInfo(clip, sourcee)
	set_output(clip,sourcee)
