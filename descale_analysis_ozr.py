from vstools import core, vs, depth, get_depth, get_y, get_w, join, plane, scale_value
from vskernels import KernelT
from functools import partial
from math import floor

def get_hist(rescaled: vs.VideoNode, source: vs.VideoNode) -> vs.VideoNode:
    diff = core.std.MakeDiff(source, rescaled)
    diff = core.rgsf.RemoveGrain(diff, 1)
    diff = core.std.Expr(diff, "x 32768 - 0 > x 255 - 32768 > x 255 - 32768 ? x 255 + 32768 < x 255 + 32768 ? ?")
    diff = depth(diff, 8, dither_type='none')
    diff = core.hist.Luma(diff)
    
    return diff

def descale_cropping_args(clip: vs.VideoNode, src_height: float, base_height: int, base_width: int, mode: str = 'wh') -> dict:    
    assert base_height >= src_height
    src_width = src_height * clip.width / clip.height
    cropped_width = base_width - 2 * floor((base_width - src_width) / 2)
    cropped_height = base_height - 2 * floor((base_height - src_height) / 2)
    
    args = dict(
        width = clip.width,
        height = clip.height
    )
    
    args_w = dict(
        width = cropped_width,
        src_width = src_width,
        src_left = (cropped_width - src_width) / 2
    )
    
    args_h = dict(
        height = cropped_height,
        src_height = src_height,
        src_top = (cropped_height - src_height) / 2
    )
    
    if 'w' in mode.lower():
        args.update(args_w)
    if 'h' in mode.lower():
        args.update(args_h)
        
    return args

def gen_descale_error(
    clip: vs.VideoNode, src_height: float, base_height: int, base_width: int, 
    kernel: KernelT, mode: str = 'wh', thr: float = 0.01
) -> vs.VideoNode:
    clip = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    
    cropping_args = descale_cropping_args(clip, src_height, base_height, base_width, mode)
    descaled = kernel.descale(clip, **cropping_args)
    
    cropping_args.update(width=clip.width, height=clip.height)
    rescaled = kernel.scale(descaled, **cropping_args)
    
    diff = core.std.Expr([clip, rescaled], f'x y - abs dup {thr} > swap 0 ?').std.Crop(10, 10, 10, 10)
    diff = core.std.Expr([diff], f'x 32 *')
    
    return diff

def gen_descale_error_manual(
    clip: vs.VideoNode, width: float, height: float, 
    src_top: float, src_height: float, src_width: float, src_left: float, 
    kernel: KernelT, thr: float = 0.01
) -> vs.VideoNode:
    clip = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    
    descaled = kernel.descale(clip, width=width, height=height, src_top=src_top, src_height=src_height, src_width=src_width, src_left=src_left)
    rescaled = kernel.scale(descaled, width=clip.width, height=clip.height, src_top=src_top, src_height=src_height, src_width=src_width, src_left=src_left)
    
    diff = core.std.Expr([clip, rescaled], f'x y - abs dup {thr} > swap 0 ?').std.Crop(10, 10, 10, 10)
    diff = core.std.Expr([diff], f'x 32 *')
    
    return diff

def gen_descale_error_width(
    clip: vs.VideoNode, width: float, height: float, 
    src_height: float, src_top: float, src_width: float, src_left: float, kernel: KernelT, 
    frame_no: int = 0, thr: float = 0.01
) -> vs.VideoNode:
    clip = clip.resize.Point(format=vs.GRAYS, matrix_s='709' if clip.format.color_family == vs.RGB else None)
    
    def _rescale(n, clip):
        descaled = kernel.descale(clip, width=width, height=height, src_height=src_height, src_top=src_top, src_left=src_left, src_width=src_width)
        
        return kernel.scale(descaled, width=clip.width, height=clip.height, src_height=src_height, src_top=src_top, src_left=src_left, src_width=src_width)
    
    rescaled = core.std.FrameEval(clip, partial(_rescale, clip=clip))
    diff = core.std.Expr([clip, rescaled], f'x y - abs dup {thr} > swap 0 ?').std.Crop(10, 10, 10, 10)
    diff = core.std.Expr([diff], f'x 32 *')
    
    return diff

def get_bad_scenes_integer(
    clip: vs.VideoNode, height: int, width: int, 
    kernel: KernelT, txt_filename: str = "encode", 
    ind_error_thr: float = 0.02, avg_error_thr: float = 0.01
):
    """
    This function returns a list of scenes that have failed a descale error test. 
    Scenes can fail if even one frame exceeds a threshold, or if the average of the scene exceeds another threshold.
    Runs before the encode starts, ideally. You should feed it the 8-bit source.
    The results will be saved to a text file so that if you have to restart the encode, you don't have to run the function over again.
    """
    
    return get_bad_scenes_fractional(clip, src_height=height, base_height=height, base_width=width, txt_filename=txt_filename, kernel=kernel, ind_error_thr=ind_error_thr, avg_error_thr=avg_error_thr)

def get_bad_scenes_fractional(
    clip: vs.VideoNode, 
    src_height: float, base_height: int, 
    base_width: int, kernel: KernelT, txt_filename: str = "encode", 
    ind_error_thr: float = 0.008, avg_error_thr: float = 0.004, thr: float = 0.01
):
    if not isinstance(txt_filename, str):
        raise TypeError("txt_filename must be a string")
    
    clipdown = core.resize.Bicubic(clip, 854, 480, format=vs.YUV420P8)
    clipdown = core.wwxd.WWXD(clipdown)
    
    comp_mask = core.std.Sobel(clip, [0])
    comp_mask = core.std.ShufflePlanes(comp_mask, 0, vs.GRAY)
    comp_mask = core.std.PlaneStats(comp_mask, prop='PS')
    
    diff = gen_descale_error(clip, src_height=src_height, base_height=base_height, base_width=base_width, kernel=kernel, thr=thr)
    diff = core.std.PlaneStats(diff, prop='PS')
    
    the_string = ""
    defective = 0
    frames = 0
    total_error = 0
    start = 0
    end = 0
    not_catalogued = 0
    
    for n in range(len(clip)):
        if n % 100 == 0:
            print(n)
            
        if (clipdown.get_frame(n).props.Scenechange == 1) and (n != 0):
            avg_error = total_error / frames
            
            if avg_error > avg_error_thr:
                defective = 1
                print(n, avg_error)
                
            total_error = 0
            frames = 0
            
            if defective == 0:
                if not_catalogued == 1:
                    the_string = the_string + f"[{start} {end}] "
                    print(the_string)
                    not_catalogued = 0
                start = n
            else:
                end = n - 1
                defective = 0
                not_catalogued = 1
                
        if (n == len(clip) - 1):
            avg_error = total_error / frames
            
            if avg_error > avg_error_thr:
                defective = 1
                
            if defective == 1:
                end = n
                the_string = the_string + f"[{start} {end}] "
            elif not_catalogued == 1:
                the_string = the_string + f"[{start} {end}] "
                
        if defective == 1:
            frames += 1
            continue
        
        mask_value = comp_mask.get_frame(n).props.PSAverage
        diff_value = diff.get_frame(n).props.PSAverage
        
        if mask_value == 0:
            diff_primary = 0
        else:
            diff_primary = diff_value / mask_value
            
        frames += 1
        total_error += diff_primary
        
        if diff_primary > ind_error_thr:
            print(n)
            defective = 1
            
    with open(f"{txt_filename}.txt", "w") as x:
        x.write(the_string)
        
    return the_string

def get_bad_scenes_manual(
    clip: vs.VideoNode, height: float, width: float, 
    src_top: float, src_height: float, src_width: float, src_left: float, 
    kernel: KernelT, txt_filename: str = "encode", 
    ind_error_thr: float = 0.008, avg_error_thr: float = 0.004, thr: float = 0.01
):
    if not isinstance(txt_filename, str):
        raise TypeError("txt_filename must be a string")
    
    clipdown = core.resize.Bicubic(clip, 854, 480, format=vs.YUV420P8)
    clipdown = core.wwxd.WWXD(clipdown)
    
    comp_mask = core.std.Sobel(clip, [0])
    comp_mask = core.std.ShufflePlanes(comp_mask, 0, vs.GRAY)
    comp_mask = core.std.PlaneStats(comp_mask, prop='PS')
    
    diff = gen_descale_error_manual(clip, width=width, height=height, src_top=src_top, src_height=src_height, src_width=src_width, src_left=src_left, kernel=kernel, thr=thr)
    diff = core.std.PlaneStats(diff, prop='PS')
    
    the_string = ""
    defective = 0
    frames = 0
    total_error = 0
    start = 0
    end = 0
    not_catalogued = 0
    
    for n in range(len(clip)):
        if n % 100 == 0:
            print(n)
            
        if (clipdown.get_frame(n).props.Scenechange == 1) and (n != 0):
            avg_error = total_error / frames
            
            if avg_error > avg_error_thr:
                defective = 1
                print(n, avg_error)
                
            total_error = 0
            frames = 0
            
            if defective == 0:
                if not_catalogued == 1:
                    the_string = the_string + f"[{start} {end}] "
                    print(the_string)
                    not_catalogued = 0
                    
                start = n
            else:
                end = n - 1
                defective = 0
                not_catalogued = 1
                
        if (n == len(clip) - 1):
            avg_error = total_error / frames
            
            if avg_error > avg_error_thr:
                defective = 1
                
            if defective == 1:
                end = n
                the_string = the_string + f"[{start} {end}] "
            elif not_catalogued == 1:
                the_string = the_string + f"[{start} {end}] "
                
        if defective == 1:
            frames += 1
            continue
        
        mask_value = comp_mask.get_frame(n).props.PSAverage
        diff_value = diff.get_frame(n).props.PSAverage
        
        if mask_value == 0:
            diff_primary = 0
        else:
            diff_primary = diff_value / mask_value
            
        frames += 1
        total_error += diff_primary
        
        if diff_primary > ind_error_thr:
            print(n)
            defective = 1
            
    with open(f"{txt_filename}.txt", "w") as x:
        x.write(the_string)
        
    return the_string

def arbitrary_kernels_fractional(
    clip: vs.VideoNode, txt_filename: str, 
    targets: list[tuple[KernelT, float, int, int, float | None, float | None, float | None]], 
    ind_error_thr: float = 0.01, avg_error_thr: float = 0.006, 
    exclude_ranges = None
):
    """target format: (kernel, height, base_height, base_width, bias, ind_error_ker, avg_error_ker)"""
    
    if not isinstance(txt_filename, str):
        raise TypeError("txt_filename must be a string")
    
    clipdown = core.resize.Bicubic(clip, 854, 480, format=vs.YUV420P8)
    clipdown = core.wwxd.WWXD(clipdown)
    
    #sw = clip.width
    #sh = clip.height
    kernel_appends = []
    kernel_diffs = []
    frame_strings = []
    defective = []
    total_error = []
    start = []
    end = []
    not_catalogued = []
    
    for target in targets:
        kernel = target[0]
        kerstr = kernel.__class__.__name__.lower()

        height = target[1]
        base_height = target[2]
        base_width = target[3]
        
        bias = target[4]
        ind_error_ker = target[5]
        avg_error_ker = target[6]
        
        if not isinstance(base_height, int):
            Exception("base_height must be an int")
            
        if not isinstance(base_width, int):
            Exception("base_width must be an int")
            
        kernel_append = kerstr
        
        if kerstr == "bicubic":
            kernel_append += f"_{kernel.b}_{kernel.c}"
        elif kerstr == "lanczos":
            kernel_append += f"_{kernel.taps}"
            
        if height == None:
            kernel_append += f"_{base_height}"
        else:
            kernel_append += f"_{height}"
            
        kernel_appends.append(kernel_append)
        
        if height == None:
            diff = gen_descale_error_manual(clip, width=base_width, height=base_height, src_top=0, src_height=base_height, src_width=base_width, src_left=0, kernel=kernel)
        else:
            diff = gen_descale_error(clip, src_height=height, base_height=base_height, base_width=base_width, kernel=kernel)
            
        diff = core.std.PlaneStats(diff, prop='PS')
        
        kernel_diffs.append(diff)
        frame_strings.append("")
        defective.append(0)
        total_error.append(0)
        start.append(0)
        end.append(0)
        not_catalogued.append(False)
        
    nokernel_string = ""
    nokernel_start = 0
    nokernel_end = 0
    nokernel_not_catalogued = False
    
    exclude = []
    if exclude_ranges:
        for thing1 in exclude_ranges:
            for thing2 in thing1:
                exclude.append(thing2)
                
    comp_mask = core.std.Sobel(clip, [0])
    comp_mask = core.std.ShufflePlanes(comp_mask, 0, vs.GRAY)
    comp_mask = core.std.PlaneStats(comp_mask, prop='PS')
    
    frames = 0
    
    for n in range(len(clip)):
        if n % 100 == 0:
            print(n)
            
        if (clipdown.get_frame(n).props.Scenechange == 1) and (n != 0):
            avg_error = []
            
            for m in range(len(total_error)):
                bias = targets[m][4]
                
                if bias == None:
                    bias = 1
                    
                avg_error.append(total_error[m]/frames/bias)
                
            lowest_error = min(avg_error)
            all_defective = True
            matches_lowest = 0
            
            for m in range(len(total_error)):
                if targets[m][6] != None:
                    avg_error_temp = targets[m][6]
                else:
                    avg_error_temp = avg_error_thr
                    
                if avg_error[m] != lowest_error or avg_error[m] > avg_error_temp:
                    defective[m] = 1
                    
                if avg_error[m] == lowest_error:
                    matches_lowest += 1
                    
                if defective[m] == 0:
                    all_defective = False
                    
            if matches_lowest > 1:
                for m in range(len(total_error)):
                    defective[m] = 1
                    
            for m in range(len(total_error)):
                total_error[m] = 0
                
                if defective[m] == 0:
                    if not_catalogued[m] == 1:
                        frame_strings[m] = frame_strings[m] + f"[{start[m]} {end[m]}] "
                        print(f"{kernel_appends[m]} is {frame_strings[m]}")
                        not_catalogued[m] = 0
                        
                    start[m] = n
                else:
                    end[m] = n - 1
                    defective[m] = 0
                    not_catalogued[m] = 1
                    
            if all_defective:
                if nokernel_not_catalogued == 1:
                    nokernel_string = nokernel_string + f"[{nokernel_start} {nokernel_end}] "
                    print(f"no-kernel is {nokernel_string}")
                    nokernel_not_catalogued = 0
                    
                nokernel_start = n
            else:
                nokernel_end = n - 1
                nokernel_not_catalogued = 1
                
            frames = 0
            
        if (n == len(clip) - 1):
            avg_error = []
            
            for m in range(len(total_error)):
                bias = targets[m][4]
                
                if bias == None:
                    bias = 1
                    
                avg_error.append(total_error[m]/frames/bias)
                
            lowest_error = min(avg_error)
            
            for m in range(len(total_error)):
                if targets[m][6] != None:
                    avg_error_temp = targets[m][6]
                else:
                    avg_error_temp = avg_error_thr
                    
                if avg_error[m] != lowest_error or avg_error[m] > avg_error_temp:
                    defective[m] = 1
                    
                if defective[m] == 1:
                    end[m] = n
                    frame_strings[m] = frame_strings[m] + f"[{start[m]} {end[m]}] "
                elif not_catalogued[m] == 1:
                    frame_strings[m] = frame_strings[m] + f"[{start[m]} {end[m]}] "
                    
        skip = True
        
        for m in range(len(total_error)):
            if defective[m] == 0:
                skip = False
                
        if skip == True or n in exclude:
            frames += 1
            continue
        
        mask_value = comp_mask.get_frame(n).props.PSAverage
        
        for m in range(len(total_error)):
            diff_value = kernel_diffs[m].get_frame(n).props.PSAverage
            
            if mask_value == 0:
                diff_primary = 0
            else:
                diff_primary = diff_value / mask_value
                
            total_error[m] = total_error[m] + diff_primary
            
            if targets[m][5] != None:
                ind_error_thr_temp = targets[m][5]
            else:
                ind_error_thr_temp = ind_error_thr
                
            if diff_primary > ind_error_thr_temp:
                defective[m] = 1
                
        frames += 1
        
    for m in range(len(total_error)):
        with open(f"{txt_filename}_{kernel_appends[m]}.txt", "w") as x:
            x.write(frame_strings[m])
            
    with open(f"{txt_filename}_nokernel.txt", "w") as x:
        x.write(nokernel_string)

def arbitrary_kernels_manual(
    clip: vs.VideoNode, txt_filename: str, 
    targets: list[tuple[KernelT, tuple[float, float, float, float], int, int, float | None, float | None, float | None]], 
    ind_error_thr: float = 0.01, avg_error_thr: float = 0.006, 
    exclude_ranges = None
):
    """
    target format: (kernel, src_, base_height, base_width, bias, ind_error_ker, avg_error_ker)\n
    src_ format: (src_top, src_height, src_left, src_width)
    """
    
    if not isinstance(txt_filename, str):
        raise TypeError("txt_filename must be a string")
    
    clipdown = core.resize.Bicubic(clip, 854, 480, format=vs.YUV420P8)
    clipdown = core.wwxd.WWXD(clipdown)
    
    #sw = clip.width
    #sh = clip.height
    kernel_appends = []
    kernel_diffs = []
    frame_strings = []
    defective = []
    total_error = []
    start = []
    end = []
    not_catalogued = []
    
    for target in targets:
        kernel = target[0]
        kerstr = kernel.__class__.__name__.lower()
        
        srcs = target[1]
        src_top = srcs[0]
        src_height = srcs[1]
        src_left = srcs[2]
        src_width = srcs[3]
        
        base_height = target[2]
        base_width = target[3]
        bias = target[4]
        ind_error_ker = target[5]
        avg_error_ker = target[6]
        
        if not isinstance(base_height, int):
            Exception("base height must be an int")
            
        if not isinstance(base_width, int):
            Exception("base width must be an int")
            
            
        kernel_append = kerstr
        
        if kerstr == "bicubic":
            kernel_append += f"_{kernel.b}_{kernel.c}"
        elif kerstr == "lanczos":
            kernel_append += f"_{kernel.taps}"
            
        kernel_append += f"_{base_height}_{base_width}_{src_top}_{src_height}_{src_left}_{src_width}"
        kernel_appends.append(kernel_append)
        
        diff = gen_descale_error_manual(clip, width=base_width, height=base_height, src_top=src_top, src_height=src_height, src_width=src_width, src_left=src_left, kernel=kernel)
        diff = core.std.PlaneStats(diff, prop='PS')
        
        kernel_diffs.append(diff)
        frame_strings.append("")
        defective.append(0)
        total_error.append(0)
        start.append(0)
        end.append(0)
        not_catalogued.append(False)
        
    nokernel_string = ""
    nokernel_start = 0
    nokernel_end = 0
    nokernel_not_catalogued = False
    
    exclude = []
    if exclude_ranges:
        for thing1 in exclude_ranges:
            for thing2 in thing1:
                exclude.append(thing2)
                
    comp_mask = core.std.Sobel(clip, [0])
    comp_mask = core.std.ShufflePlanes(comp_mask, 0, vs.GRAY)
    comp_mask = core.std.PlaneStats(comp_mask, prop='PS')
    
    frames = 0
    
    for n in range(len(clip)):
        if n % 100 == 0:
            print(n)
            
        if (clipdown.get_frame(n).props.Scenechange == 1) and (n != 0):
            avg_error = []
            
            for m in range(len(total_error)):
                bias = targets[m][4]
                
                if bias == None:
                    bias = 1
                    
                avg_error.append(total_error[m]/frames/bias)
                
            lowest_error = min(avg_error)
            all_defective = True
            matches_lowest = 0
            
            for m in range(len(total_error)):
                if targets[m][6] != None:
                    avg_error_temp = targets[m][6]
                else:
                    avg_error_temp = avg_error_thr
                    
                if avg_error[m] != lowest_error or avg_error[m] > avg_error_temp:
                    defective[m] = 1
                    
                if avg_error[m] == lowest_error:
                    matches_lowest += 1
                    
                if defective[m] == 0:
                    all_defective = False
                    
            if matches_lowest > 1:
                for m in range(len(total_error)):
                    defective[m] = 1
                    
            for m in range(len(total_error)):
                total_error[m] = 0
                
                if defective[m] == 0:
                    if not_catalogued[m] == 1:
                        frame_strings[m] = frame_strings[m] + f"[{start[m]} {end[m]}] "
                        print(f"{kernel_appends[m]} is {frame_strings[m]}")
                        not_catalogued[m] = 0
                        
                    start[m] = n
                else:
                    end[m] = n - 1
                    defective[m] = 0
                    not_catalogued[m] = 1
                    
            if all_defective:
                if nokernel_not_catalogued == 1:
                    nokernel_string = nokernel_string + f"[{nokernel_start} {nokernel_end}] "
                    print(f"no-kernel is {nokernel_string}")
                    nokernel_not_catalogued = 0
                    
                nokernel_start = n
            else:
                nokernel_end = n - 1
                nokernel_not_catalogued = 1
                
            frames = 0
            
        if (n == len(clip) - 1):
            avg_error = []
            
            for m in range(len(total_error)):
                bias = targets[m][4]
                
                if bias == None:
                    bias = 1
                    
                avg_error.append(total_error[m]/frames/bias)
                
            lowest_error = min(avg_error)
            
            for m in range(len(total_error)):
                if targets[m][6] != None:
                    avg_error_temp = targets[m][6]
                else:
                    avg_error_temp = avg_error_thr
                    
                if avg_error[m] != lowest_error or avg_error[m] > avg_error_temp:
                    defective[m] = 1
                    
                if defective[m] == 1:
                    end[m] = n
                    frame_strings[m] = frame_strings[m] + f"[{start[m]} {end[m]}] "
                elif not_catalogued[m] == 1:
                    frame_strings[m] = frame_strings[m] + f"[{start[m]} {end[m]}] "
        
        skip = True
        
        for m in range(len(total_error)):
            if defective[m] == 0:
                skip = False
                
        if skip == True or n in exclude:
            frames += 1
            continue
        
        mask_value = comp_mask.get_frame(n).props.PSAverage
        
        for m in range(len(total_error)):
            diff_value = kernel_diffs[m].get_frame(n).props.PSAverage
            
            if mask_value == 0:
                diff_primary = 0
            else:
                diff_primary = diff_value / mask_value
                
            total_error[m] = total_error[m] + diff_primary
            
            if targets[m][5] != None:
                ind_error_thr_temp = targets[m][5]
            else:
                ind_error_thr_temp = ind_error_thr
                
            if diff_primary > ind_error_thr_temp:
                defective[m] = 1
                
        frames += 1
        
    for m in range(len(total_error)):
        with open(f"{txt_filename}_{kernel_appends[m]}.txt", "w") as x:
            x.write(frame_strings[m])
            
    with open(f"{txt_filename}_nokernel.txt", "w") as x:
        x.write(nokernel_string)

def choose_luma(
    clip_main: vs.VideoNode, clip_alt: vs.VideoNode, txt_filename: str, 
    kernel: KernelT, src_height: float, base_height: int, base_width: int, 
    clip_main_name: str = "clip1", clip_alt_name: str = "clip2",
    source_1_bias: float = 1, exclude_ranges = None, dont_care_thr: float = 0.001
):
    if len(clip_main) != len(clip_alt):
        Exception("Both clips need to be the same length")
        
    if not isinstance(txt_filename, str):
        raise TypeError("txt_filename must be a string")
    
    clipdown = core.resize.Bicubic(clip_main, 854, 480)
    clipdown = core.wwxd.WWXD(clipdown)
    exclude = []
    
    if exclude_ranges:
        for thing1 in exclude_ranges:
            for thing2 in thing1:
                exclude.append(thing2)
    
    comp_mask1 = core.std.Sobel(clip_main, [0])
    comp_mask1 = core.std.ShufflePlanes(comp_mask1, 0, vs.GRAY)
    comp_mask1 = core.std.PlaneStats(comp_mask1, prop='PS')
    
    comp_mask2 = core.std.Sobel(clip_alt, [0])
    comp_mask2 = core.std.ShufflePlanes(comp_mask2, 0, vs.GRAY)
    comp_mask2 = core.std.PlaneStats(comp_mask2, prop='PS')
    
    diff_847 = gen_descale_error(clip_main, src_height=src_height, base_height=base_height, base_width=base_width, kernel=kernel)
    diff_844 = gen_descale_error(clip_alt, src_height=src_height, base_height=base_height, base_width=base_width, kernel=kernel)
    
    diff_847 = core.std.PlaneStats(diff_847, prop='PS')
    diff_844 = core.std.PlaneStats(diff_844, prop='PS')
    
    the_string_847 = ""
    defective_847 = 0
    frames = 0
    total_error_847 = 0
    start_847 = 0
    end_847 = 0
    not_catalogued_847 = 0
    the_string_844 = ""
    defective_844 = 0
    total_error_844 = 0
    start_844 = 0
    end_844 = 0
    not_catalogued_844 = 0
    
    for n in range(len(clip_main)):
        if n % 100 == 0:
            print(n)
            
        if (clipdown.get_frame(n).props.Scenechange == 1) and (n != 0):
            avg_error_847 = total_error_847 / frames
            avg_error_844 = total_error_844 / frames
            
            print(avg_error_847)
            print(avg_error_844)
            
            if avg_error_847 > avg_error_844 * source_1_bias and avg_error_847 > dont_care_thr and defective_844 != 1:
                defective_847 = 1
            else:
                defective_844 = 1
                
            total_error_847 = 0
            
            if defective_847 == 0:
                if not_catalogued_847 == 1:
                    the_string_847 = the_string_847 + f"[{start_847} {end_847}] "
                    print(f"{clip_alt_name} is {the_string_847}")
                    not_catalogued_847 = 0
                    
                start_847 = n
            else:
                end_847 = n - 1
                defective_847 = 0
                not_catalogued_847 = 1
                
            total_error_844 = 0
            frames = 0
            
            if defective_844 == 0:
                if not_catalogued_844 == 1:
                    the_string_844 = the_string_844 + f"[{start_844} {end_844}] "
                    print(f"{clip_main_name} is {the_string_844}")
                    not_catalogued_844 = 0
                    
                start_844 = n
            else:
                end_844 = n - 1
                defective_844 = 0
                not_catalogued_844 = 1
                
        if (n == len(clip_main) - 1):
            avg_error_847 = total_error_847 / frames
            
            if defective_847 == 1:
                end_847 = n
                the_string_847 = the_string_847 + f"[{start_847} {end_847}] "
            elif not_catalogued_847 == 1:
                the_string_847 = the_string_847 + f"[{start_847} {end_847}] "
                
            avg_error_844 = total_error_844 / frames
            
            if defective_844 == 1:
                end_844 = n
                the_string_844 = the_string_844 + f"[{start_844} {end_844}] "
            elif not_catalogued_844 == 1:
                the_string_844 = the_string_844 + f"[{start_844} {end_844}] "
                
        if n in exclude or defective_844 == 1:
            frames += 1
            continue
        
        mask_value1 = comp_mask1.get_frame(n).props.PSAverage
        mask_value2 = comp_mask2.get_frame(n).props.PSAverage
        diff_value_847 = diff_847.get_frame(n).props.PSAverage
        
        if mask_value1 == 0:
            diff_primary_847 = 0
        else:
            diff_primary_847 = diff_value_847 / mask_value1
            
        total_error_847 += diff_primary_847
        diff_value_844 = diff_844.get_frame(n).props.PSAverage
        
        if mask_value2 == 0:
            diff_primary_844 = 0
        else:
            diff_primary_844 = diff_value_844 / mask_value2
            
        if diff_primary_844 > diff_primary_847 * 1.5:
            defective_844 = 1
            
        frames += 1
        total_error_844 += diff_primary_844
        
    with open(f"{txt_filename}_{clip_alt_name}.txt", "w") as x:
        x.write(the_string_847)
        
    return the_string_847, the_string_844

def test_descale_error_fractional(
    clip: vs.VideoNode, src_height: float, 
    base_height: int, base_width: int, kernel: KernelT
):
    def get_calc(n, f, clip, core):
        diff_raw = f[0].props['PSAverage']
        mask_value = f[1].props['PSAverage']
        
        if mask_value == 0:
            diff_primary = 0
        else:
            diff_primary = diff_raw / mask_value
            
        return core.text.Text(clip, str(diff_primary))
    
    #sw = clip.width
    #sh = clip.height
    comp_mask = core.std.Sobel(clip, [0])
    comp_mask = core.std.ShufflePlanes(comp_mask, 0, vs.GRAY)
    comp_mask = core.std.PlaneStats(comp_mask, prop='PS')
    
    diff = gen_descale_error(clip, src_height=src_height, base_height=base_height, base_width=base_width, kernel=kernel)
    diff = core.std.PlaneStats(diff, prop='PS')
    
    return core.std.FrameEval(diff, partial(get_calc, clip=diff, core=vs.core), prop_src=[diff, comp_mask])

def test_descale_error_manual(
    clip: vs.VideoNode, height: float, width: float, 
    src_top: float, src_height: float, src_width: float, src_left: float, kernel: KernelT
):
    def get_calc(n, f, clip, core):
        diff_raw = f[0].props['PSAverage']
        mask_value = f[1].props['PSAverage']
        
        if mask_value == 0:
            diff_primary = 0
        else:
            diff_primary = diff_raw / mask_value
            
        return core.text.Text(clip, str(diff_primary))
    
    #sw = clip.width
    #sh = clip.height
    comp_mask = core.std.Sobel(clip, [0])
    comp_mask = core.std.ShufflePlanes(comp_mask, 0, vs.GRAY)
    comp_mask = core.std.PlaneStats(comp_mask, prop='PS')
    
    diff = gen_descale_error_manual(clip, width=width, height=height, src_top=src_top, src_height=src_height, src_width=src_width, src_left=src_left, kernel=kernel)
    diff = core.std.PlaneStats(diff, prop='PS')
    
    return core.std.FrameEval(diff, partial(get_calc, clip=diff, core=vs.core), prop_src=[diff, comp_mask])

def test_descale_error_integer(clip: vs.VideoNode, width: int, height: int, kernel: KernelT):
    return test_descale_error_fractional(clip, src_height=height, base_height=height, base_width=width, kernel=kernel)

import vapoursynth as vs
from vapoursynth import core
from vstools import depth
from vskernels import Bilinear,Lanczos,Bicubic

clip1 = core.lsmas.LWLibavSource(r"D:\Anime\Mashiro no Oto\[GHS] Mashiro no Oto [WEB 1080p] (Batch)\[GHS] Mashiro no Oto - 02v2 [14186932].mkv")
clip2 = core.lsmas.LWLibavSource(r"D:\Anime\Mashiro no Oto\[HLouis]Mashiro no Oto 2021.1080p.WEB-DL.H264\[HLouis]Mashiro no Oto 2021.S01E02.1080p.WEB-DL.H264.AAC.mkv")

arbitrary_kernels_fractional(clip1,"mnoe2cr.txt",[[Bicubic, 765.05, 1080, 1920, 1/3, 1/3, None, None, None, None],[Lanczos, 765.05, 1080, 1920, None, None, 4, None, None, None],[Lanczos, 765.05, 1080, 1920, None, None, 3, None, None, None],[Bilinear, 765.05, 1080, 1920, None,None,None, None, None, None]])