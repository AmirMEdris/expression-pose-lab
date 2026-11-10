"""Local expression and pose demo. Run --help for image, video, and camera modes."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from backend import select_backend
select_backend()

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import cv2
from pose import PoseEstimator, draw_poses
from pose_quality import assess_pose
from camera import open_camera

ROOT = Path(__file__).resolve().parent


def annotate(frame, poses, faces, show_partial=False, landmark_threshold=.5):
    result = frame.copy()
    usable = [p for p in poses if assess_pose(p, landmark_threshold)['displayable']]
    draw_poses(result, poses if show_partial else usable, visibility=landmark_threshold)
    for face in faces:
        x,y,w,h = face['box']
        cv2.rectangle(result, (x,y), (x+w,y+h), (80,230,80), 2)
        # Keep overlays legible in groups; full uncalibrated scores live in JSON.
        cv2.putText(result, face['label'], (x,max(16,y-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (80,230,80), 2)
    if show_partial:
        message = 'Diagnostic: partial poses may be wrong'
    elif len(usable) < len(poses):
        message = f'Withheld {len(poses)-len(usable)} incomplete pose estimate(s)'
    elif not poses:
        message = 'No pose detected'
    else:
        message = 'Estimated pose - alignment not verified'
    cv2.rectangle(result, (0,result.shape[0]-27), (result.shape[1],result.shape[0]), (25,25,25), -1)
    cv2.putText(result,message,(8,result.shape[0]-9),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--video', type=Path)
    source.add_argument('--camera', type=int)
    source.add_argument('--screen', action='store_true', help='Start interactive screen capture; A switches to camera')
    parser.add_argument('--screen-region', type=int, nargs=4, metavar=('LEFT','TOP','WIDTH','HEIGHT'))
    parser.add_argument('--capture-dir',type=Path,default=Path('results/captures'))
    parser.add_argument('--output', type=Path, help='PNG/JPG for images, MP4 for video; also saves JSON/JSONL')
    parser.add_argument('--pose-only', action='store_true')
    parser.add_argument('--pose-model', choices=('lite','full','heavy'), default='lite')
    parser.add_argument('--show-partial-poses', action='store_true', help='Diagnostic: draw incomplete pose estimates, which may be wrong')
    parser.add_argument('--landmark-threshold', type=float, default=.5, help='Minimum visibility and presence for major joints')
    parser.add_argument('--max-people', type=int, default=4)
    parser.add_argument('--expression-threshold', type=float, default=.6)
    parser.add_argument('--max-frames', type=int, default=0, help='0 means unlimited')
    parser.add_argument('--show', action='store_true', help='Display window; press Q or Esc to exit')
    args = parser.parse_args()
    if args.screen_region and (args.screen_region[2]<=0 or args.screen_region[3]<=0):
        parser.error('Screen region width and height must be positive')
    if args.max_people < 1 or args.max_frames < 0 or not 0 <= args.expression_threshold <= 1 or not 0 <= args.landmark_threshold <= 1:
        parser.error('Invalid people, frame limit, or threshold value')
    video_mode = args.image is None
    if args.output and args.output.suffix.lower() not in (('.mp4',) if video_mode else ('.png','.jpg','.jpeg')):
        parser.error('Use .mp4 output for streams or .png/.jpg output for images')
    models = ROOT / 'models'
    needed = [models / f'pose_landmarker_{args.pose_model}.task']
    if not args.pose_only:
        needed.append(models / 'expression-baseline.h5')
    if any(not p.is_file() for p in needed):
        parser.error(f'Model files missing. Run: python download_models.py --pose-model {args.pose_model}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.camera is not None or args.screen:
        from live import run_live
        run_live(args,ROOT)
        return
    pose = PoseEstimator(needed[0], video=video_mode, max_people=args.max_people)
    capture = writer = stream = None
    try:
        expression = None
        if not args.pose_only:
            from expression import ExpressionEstimator
            expression = ExpressionEstimator(needed[1], args.expression_threshold)
        if not video_mode:
            frame = cv2.imread(str(args.image))
            if frame is None:
                parser.error(f'Cannot read image: {args.image}')
            poses = pose.detect(frame)
            faces = expression.detect(frame) if expression else []
            output = args.output or Path('results/image.png')
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), annotate(frame, poses, faces, args.show_partial_poses, args.landmark_threshold)):
                raise OSError(f'Cannot write {output}')
            report = {'expression_model':'recovered baseline' if expression else None,
                      'pose_model':args.pose_model, 'show_partial_poses':args.show_partial_poses,
                      'landmark_threshold':args.landmark_threshold,
                      'pose_quality':[assess_pose(p,args.landmark_threshold) for p in poses],
                      'poses':poses, 'faces':faces}
            output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            drawn = len(poses) if args.show_partial_poses else sum(q['displayable'] for q in report['pose_quality'])
            print(f'Saved {output}: {len(poses)} detected / {drawn} drawn poses, {len(faces)} faces')
            return
        first_frame = None
        if args.camera is not None:
            capture, first_frame = open_camera(args.camera)
        else:
            capture = cv2.VideoCapture(str(args.video))
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open video: {args.video}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        if args.output:
            stream = args.output.with_suffix('.jsonl').open('w', encoding='utf-8')
        index = 0
        started = time.monotonic()
        previous_timestamp = -1
        while args.max_frames == 0 or index < args.max_frames:
            if first_frame is not None:
                ok, frame = True, first_frame
                first_frame = None
            else:
                ok, frame = capture.read()
            if not ok:
                if index == 0:
                    raise RuntimeError('Input opened, but no frame could be read')
                break
            timestamp = int((time.monotonic()-started)*1000) if args.camera is not None else round(index*1000/fps)
            timestamp = max(previous_timestamp+1, timestamp)
            previous_timestamp = timestamp
            poses = pose.detect(frame, timestamp)
            faces = expression.detect(frame) if expression else []
            annotated = annotate(frame, poses, faces, args.show_partial_poses, args.landmark_threshold)
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1],frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Video writer failed to open')
                writer.write(annotated)
                stream.write(json.dumps({'frame':index,'timestamp_ms':timestamp,
                    'pose_model':args.pose_model,'show_partial_poses':args.show_partial_poses,
                    'landmark_threshold':args.landmark_threshold,
                    'pose_quality':[assess_pose(p,args.landmark_threshold) for p in poses],
                    'poses':poses,'faces':faces})+'\n')
            index += 1
            if args.show or args.camera is not None:
                cv2.imshow('Expression + Pose Lab - Q to quit', annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'),27):
                    break
        print(f'Processed {index} frames in {time.monotonic()-started:.2f}s')
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if stream is not None:
            stream.close()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
