#!/usr/bin/env python3
"""
semantic_map_node.py  (v2)
--------------------------
Key fixes over v1:
  - Uses TF2 (map->base_link) instead of raw /odom for pose
  - Vectorised batch update (much faster)
  - Saves every 5s so web UI updates live
  - Filters ground/ceiling points to reduce noise
  - Reads camera intrinsics from /camera/camera_info automatically
  - Listens to /pointcloud/filtered first, falls back to /point_cloud2
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np
import cv2
import torch
import open_clip
import threading
import time
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MAP_RESOLUTION  = 0.10
MAP_SIZE_M      = 20.0
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EVERY_N    = 5
SAVE_DIR        = Path("/vlmaps/data")
SAVE_EVERY_S    = 5.0
MIN_Z           = -0.3
MAX_Z           =  2.5
MAX_RANGE       = 8.0

T_LIDAR_TO_CAM = np.array([
    [ 0.0, -1.0,  0.0,  0.05],
    [ 0.0,  0.0, -1.0, -0.03],
    [ 1.0,  0.0,  0.0,  0.00],
    [ 0.0,  0.0,  0.0,  1.00],
], dtype=np.float64)

_shared_map = None
_shared_map_lock = threading.Lock()

def get_shared_map():
    return _shared_map


class SemanticMap:
    def __init__(self, resolution=MAP_RESOLUTION, size_m=MAP_SIZE_M):
        self.res    = resolution
        self.N      = int(size_m / resolution)
        self.origin = np.array([size_m / 2.0, size_m / 2.0])
        self.feat_sum   = np.zeros((self.N, self.N, 512), dtype=np.float32)
        self.feat_count = np.zeros((self.N, self.N),      dtype=np.int32)
        self.rgb_sum    = np.zeros((self.N, self.N, 3),   dtype=np.float32)
        self.rgb_count  = np.zeros((self.N, self.N),      dtype=np.int32)
        self.lock       = threading.Lock()

    def world_to_cell(self, xy):
        c = ((xy + self.origin) / self.res).astype(int)
        if c.ndim == 1:
            return c if (0 <= c[0] < self.N and 0 <= c[1] < self.N) else None
        mask = (c[:,0] >= 0) & (c[:,0] < self.N) & (c[:,1] >= 0) & (c[:,1] < self.N)
        return c, mask

    def update_batch(self, world_xy, colors_rgb, clip_feat=None):
        cells, mask = self.world_to_cell(world_xy)
        if not mask.any():
            return
        cells  = cells[mask]
        colors = colors_rgb[mask]
        rs, cs = cells[:,0], cells[:,1]
        with self.lock:
            np.add.at(self.rgb_sum,   (rs, cs, 0), colors[:,0].astype(np.float32))
            np.add.at(self.rgb_sum,   (rs, cs, 1), colors[:,1].astype(np.float32))
            np.add.at(self.rgb_sum,   (rs, cs, 2), colors[:,2].astype(np.float32))
            np.add.at(self.rgb_count, (rs, cs),    1)
            if clip_feat is not None:
                feat = clip_feat.astype(np.float32)
                unique_rc = np.unique(np.stack([rs, cs], axis=1), axis=0)
                self.feat_sum[unique_rc[:,0], unique_rc[:,1]]   += feat
                self.feat_count[unique_rc[:,0], unique_rc[:,1]] += 1

    def get_feature_map(self):
        with self.lock:
            count = self.feat_count[:,:,None].clip(1)
            avg   = self.feat_sum / count
        norms = np.linalg.norm(avg, axis=-1, keepdims=True).clip(1e-6)
        return avg / norms

    def get_rgb_map(self):
        with self.lock:
            count = self.rgb_count[:,:,None].clip(1)
            return (self.rgb_sum / count).clip(0,255).astype(np.uint8)

    def query(self, text_feat, top_k=5):
        grid = self.get_feature_map()
        sim  = (grid @ text_feat).astype(np.float32)
        flat = np.argsort(sim.ravel())[-top_k:][::-1]
        rows, cols = np.unravel_index(flat, sim.shape)
        scores = sim[rows, cols]
        world  = np.stack([rows,cols],axis=1).astype(float)*self.res - self.origin
        return [(float(world[i,0]),float(world[i,1]),float(scores[i])) for i in range(top_k)], sim

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with self.lock:
            np.save(path/"feat_sum.npy",   self.feat_sum)
            np.save(path/"feat_count.npy", self.feat_count)
            np.save(path/"rgb_sum.npy",    self.rgb_sum)
            np.save(path/"rgb_count.npy",  self.rgb_count)
        meta = {"resolution":self.res,"size_cells":self.N,"origin":self.origin.tolist()}
        with open(path/"meta.json","w") as f:
            json.dump(meta, f)

    def load(self, path: Path):
        with open(path/"meta.json") as f:
            meta = json.load(f)
        self.res    = meta["resolution"]
        self.N      = meta["size_cells"]
        self.origin = np.array(meta["origin"])
        with self.lock:
            self.feat_sum   = np.load(path/"feat_sum.npy")
            self.feat_count = np.load(path/"feat_count.npy")
            self.rgb_sum    = np.load(path/"rgb_sum.npy")
            self.rgb_count  = np.load(path/"rgb_count.npy")


def parse_pointcloud2(msg):
    fields = {f.name: f.offset for f in msg.fields}
    if not all(k in fields for k in ('x','y','z')):
        return None
    fmt  = msg.point_step
    n    = msg.width * msg.height
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, fmt)
    xs = data[:, fields['x']:fields['x']+4].view(np.float32).flatten()
    ys = data[:, fields['y']:fields['y']+4].view(np.float32).flatten()
    zs = data[:, fields['z']:fields['z']+4].view(np.float32).flatten()
    pts = np.stack([xs,ys,zs],axis=1)
    valid = (np.isfinite(pts).all(axis=1) &
             (pts[:,2] > MIN_Z) & (pts[:,2] < MAX_Z) &
             (np.linalg.norm(pts,axis=1) > 0.15) &
             (np.linalg.norm(pts,axis=1) < MAX_RANGE))
    return pts[valid]


def tf_to_matrix(tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    x,y,z,w = q.x,q.y,q.z,q.w
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def project_to_image(pts_lidar, T, K, W, H):
    pts_h   = np.hstack([pts_lidar, np.ones((len(pts_lidar),1))])
    pts_cam = (T @ pts_h.T).T[:,:3]
    mask    = pts_cam[:,2] > 0.05
    pts_cam = pts_cam[mask]
    px = (K[0,0]*pts_cam[:,0]/pts_cam[:,2] + K[0,2]).astype(int)
    py = (K[1,1]*pts_cam[:,1]/pts_cam[:,2] + K[1,2]).astype(int)
    in_frame = (px>=0)&(px<W)&(py>=0)&(py<H)
    return px[in_frame], py[in_frame], np.where(mask)[0][in_frame]


class SemanticMapNode(Node):
    def __init__(self):
        super().__init__("semantic_map_node")
        global _shared_map

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIP on {self.device}...")
        self.clip_model, _, self.clip_prep = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device)
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.get_logger().info("CLIP ready.")

        self.K = np.array([[554.3,0,320],[0,554.3,240],[0,0,1]], dtype=np.float64)
        self.img_w = 640
        self.img_h = 480
        self.got_camera_info = False

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sem_map = SemanticMap()
        _shared_map  = self.sem_map
        if (SAVE_DIR/"meta.json").exists():
            self.sem_map.load(SAVE_DIR)
            self.get_logger().info("Resumed map from disk.")

        self.latest_image = None
        self.latest_cloud = None
        self.frame_count  = 0
        self.last_save    = time.time()
        self._img_lock    = threading.Lock()
        self._cloud_lock  = threading.Lock()

        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image,       "/camera/image_raw",    self.image_cb,   be)
        self.create_subscription(CameraInfo,  "/camera/camera_info",  self.caminfo_cb, be)
        self.create_subscription(PointCloud2, "/pointcloud/filtered", self.cloud_cb,   be)
        self.create_subscription(PointCloud2, "/point_cloud2",        self.cloud_cb,   be)
        self.create_timer(0.15, self.fuse)

        self.get_logger().info("Ready — walk the robot around to build the map!")

    def caminfo_cb(self, msg):
        if self.got_camera_info:
            return
        K = np.array(msg.k).reshape(3,3)
        self.K = K
        self.img_w = msg.width
        self.img_h = msg.height
        self.got_camera_info = True
        self.get_logger().info(f"Camera: {msg.width}x{msg.height} fx={K[0,0]:.1f}")

    def image_cb(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8","RGB8"):
                img = arr.reshape(msg.height, msg.width, 3)
            elif msg.encoding in ("bgr8","BGR8"):
                img = arr.reshape(msg.height, msg.width, 3)[:,:,::-1].copy()
            elif "mono" in msg.encoding:
                img = np.stack([arr.reshape(msg.height,msg.width)]*3, axis=-1)
            else:
                img = arr.reshape(msg.height, msg.width, -1)[:,:,:3]
            with self._img_lock:
                self.latest_image = img
        except Exception as e:
            self.get_logger().warn(f"Image err: {e}", throttle_duration_sec=5)

    def cloud_cb(self, msg):
        pts = parse_pointcloud2(msg)
        if pts is not None and len(pts) > 10:
            with self._cloud_lock:
                self.latest_cloud = pts

    def fuse(self):
        with self._img_lock:
            image = self.latest_image
        with self._cloud_lock:
            cloud = self.latest_cloud

        if self.frame_count % 20 == 0:
            self.get_logger().info(
                f"fuse called | image={'yes' if image is not None else 'NO'} "
                f"cloud={'yes' if cloud is not None else 'NO'}"
            )

        if image is None or cloud is None:
            return

        # Get pose: prefer map->base_link, fall back to odom->base_link
        T_map_base = None
        for parent in ("map", "odom"):
            try:
                tf = self.tf_buffer.lookup_transform(
                    parent, "base_link", rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
                T_map_base = tf_to_matrix(tf)
                break
            except Exception:
                continue
        if T_map_base is None:
            return

        self.frame_count += 1

        px, py, pt_idx = project_to_image(cloud, T_LIDAR_TO_CAM, self.K, self.img_w, self.img_h)
        if len(pt_idx) == 0:
            return

        colors_rgb = image[py, px]

        pts_sel   = cloud[pt_idx]
        pts_h     = np.hstack([pts_sel, np.ones((len(pts_sel),1))])
        pts_world = (T_map_base @ pts_h.T).T[:,:2]

        clip_feat = None
        if self.frame_count % CLIP_EVERY_N == 0:
            clip_feat = self._encode_clip(image)

        self.sem_map.update_batch(pts_world, colors_rgb, clip_feat)

        now = time.time()
        if now - self.last_save > SAVE_EVERY_S:
            self.sem_map.save(SAVE_DIR)
            self.last_save = now
            obs = int((self.sem_map.feat_count > 0).sum())
            self.get_logger().info(f"Saved — {obs} observed cells")

    def _encode_clip(self, rgb_image):
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb_image)
            t   = self.clip_prep(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.clip_model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
            return f.squeeze().cpu().numpy()
        except Exception as e:
            self.get_logger().warn(f"CLIP err: {e}", throttle_duration_sec=5)
            return None

    def encode_text(self, text):
        tok = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            f = self.clip_model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze().cpu().numpy()

    def shutdown(self):
        self.sem_map.save(SAVE_DIR)
        self.get_logger().info("Map saved.")


def main():
    rclpy.init()
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
