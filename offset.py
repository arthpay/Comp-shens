from vstools import vs, core

class MeasureMethod:
    PSNR = 0
    PSNR_HVS = 1
    SSIM = 2
    SSIM_MS = 3
    DIFF = 4


def find_offset(
    ref_clip: vs.VideoNode,
    clips: list[vs.VideoNode],
    approx_offset: int = 100,
    ref_frame: int = None,
    method: MeasureMethod = MeasureMethod.SSIM
) -> list:
    """Dumb function that returns the offset between sources

    Args:
        ref_clip: Reference clip
        clips: clip or clips of sources to be synced
        approx_offset: Expected offset between sources (i.e 24 frame intro). Defaults to 100.
        ref_frame: Reference frame to use, best to use unique frames with no dupes surrounding. Defaults to None.
        method: Function used to get the difference. Defaults to MeasureMethod.SSIM.

    Returns:
        offset between ref_clip and each item in [clips]
    """

    _prop = ['psnr_y', 'psnr_hvs_y', 'float_ssim', 'float_ms_ssim', 'PlaneStatsDiff']

    if not isinstance(clips, list):
        clips = [clips]

    if ref_frame is None:
        ref_frame = int(ref_clip.num_frames / 2)

    frames = ref_frame - int(approx_offset), ref_frame + int(approx_offset)
    hw_frame = abs((frames[0] - frames[1]) // 2)

    clips = [i[frames[0]:frames[1]] for i in [ref_clip, *clips]]
    ref_clip, *clips = clips

    target_frame = ref_clip[hw_frame - 1:hw_frame] * clips[0].num_frames

    _temp = [[] for _ in range(len(clips))]
    _offsets = []

    if method == MeasureMethod.DIFF:
        clips = [core.std.PlaneStats(clip, target_frame, plane=0) for clip in clips]
    else:
        clips = [core.vmaf.Metric(clip, target_frame, feature=method) for clip in clips]

    for pos, video in enumerate(clips):
        for f in video.frames():
            _temp[pos].append(f.props.get(_prop[method]))

    for i in _temp:
        smallest = min(j for j in i if j > 0) if method == MeasureMethod.DIFF else max(i)
        position = i.index(smallest)
        _offsets.append((position - hw_frame) + 1)

    return _offsets


def find_desync_point(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    approx_offset: int = 100,
    num_parts: int = 40,
    overlap: int = 33,
    method: MeasureMethod = MeasureMethod.DIFF
):
    """dumb function that finds the desync point between sources

    Args:
        clip_a: your source dummmy
        clip_b: your other source dummy
        approx_offset: Expected offset between sources (i.e 24 frame intro). Defaults to 100.
        num_parts: how many clips to split the inputs into. Defaults to 40
        overlap: percentage of overlap for each part
        method: Function used to get the difference. Defaults to MeasureMethod.DIFF.
    """
    _prop = ['psnr_y', 'psnr_hvs_y', 'float_ssim', 'float_ms_ssim', 'PlaneStatsDiff']

    _command = []
    _conquer = []

    _offsets = []
    _desyncs = []

    # Calculate the size of each non-overlapping part
    part_size = clip_a.num_frames // num_parts
    # Calculate the size of the overlap
    overlap_size = int(part_size * (overlap / 100))
    # Initialize the list of parts with the start and end indices of the first part
    parts = [(0, part_size)]
    # Initialize the start index of the next part
    start_index = part_size - overlap_size
    # Split the remaining parts
    for i in range(1, num_parts):
        # Add the start and end indices of the next part to the list
        parts.append((start_index, start_index + part_size))
        # Update the start index of the next part
        start_index += part_size - overlap_size

    for i, v in enumerate(parts):
        start, end = v[0], v[1]

        print(f"processing sector {v}, start={start}, end={end}", end='\r')

        clips = [i[start:end] for i in (clip_a, clip_b)]

        # Pad first clip?
        # if i == 0:
        #     clips = [core.std.BlankClip(i, length=72) + i for i in clips]

        offset = find_offset(
            ref_clip=clips[0], clips=clips[1], approx_offset=approx_offset, method=method
            )

        _offsets.append(offset)

        if offset != [0] and offset not in _desyncs:
            print(f"desync found in sector {v} \n ({start}, {end}), offset={offset}")
            _desyncs.append(offset)
            _command.append(start)
            _conquer.append(end)

    if len(_desyncs) != 0:
        for j, _ in enumerate(_command):

            _temp = []
            clips = [i[_command[j]:_conquer[j]] for i in (clip_a, clip_b)]

            if method != MeasureMethod.DIFF:
                frames = (clips[0].num_frames, clips[1].num_frames)

                if frames[0] != frames[1]:
                    _index = min(frames)
                    _index = frames.index(_index)

                    clips[_index] = clips[_index] + \
                        clips[_index].std.BlankClip(length=abs(clips[0].num_frames - clips[1].num_frames))

                process = core.vmaf.Metric(*clips, feature=method)
            else:
                process = core.std.PlaneStats(*clips, plane=0)

            for f in process.frames():
                _temp.append(f.props.get(_prop[method]))

            index = max(_temp) if method == MeasureMethod.DIFF else min(j for j in _temp if j > 0)
            position = _temp.index(index)

            print(f'desync found at: {(_command[j] + position)} with an approximate offset of {_desyncs[j][0]}, clip_b at now at {(_command[j] + position) + _desyncs[j][0]}') # noqa
    else:
        print('\n', "nothing found")
