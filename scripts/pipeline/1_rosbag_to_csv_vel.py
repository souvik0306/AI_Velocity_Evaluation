#!/usr/bin/env python3

import argparse
import os
import rosbag
import pandas as pd


def twist_message_to_row(t, msg):
    # Support multiple message shapes: Twist, TwistStamped, Odometry
    def _get_linear(m):
        # geometry_msgs/TwistStamped -> m.twist.linear
        if hasattr(m, "twist"):
            inner = m.twist
            if hasattr(inner, "linear"):
                return inner.linear
            # nav_msgs/Odometry -> m.twist.twist.linear
            if hasattr(inner, "twist") and hasattr(inner.twist, "linear"):
                return inner.twist.linear
        # geometry_msgs/Twist -> m.linear
        if hasattr(m, "linear"):
            return m.linear
        raise AttributeError("message has no linear velocity field")

    linear = _get_linear(msg)
    return {
        "time": t,
        "vel_x": linear.x,
        "vel_y": linear.y,
        "vel_z": linear.z,
    }


def build_velocity_dataframe(rows):
    columns = ["time", "vel_x", "vel_y", "vel_z"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def derive_output_paths(bag_path, est_out, gt_out):
    bag_dir = os.path.dirname(bag_path)
    bag_base = os.path.splitext(os.path.basename(bag_path))[0]
    est_default = os.path.join(bag_dir, f"{bag_base}_vel_est.csv")
    gt_default = os.path.join(bag_dir, f"{bag_base}_vel_gt.csv")
    return est_out or est_default, gt_out or gt_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--est_out")
    parser.add_argument("--gt_out")
    parser.add_argument(
        "--velocity_topic",
        default="/mavros/local_position/velocity_local",
        help="Estimate velocity; uses twist/linear/{x,y,z}",
    )
    parser.add_argument(
        "--gt_velocity_topic",
        default="/vrpn_client_node/AIIMU1/twist",
        help="Ground truth velocity topic (e.g. /vrpn_client_node/AIIMU1/twist)",
    )
    
    # default="/mavros/vision_speed/speed_twist",
    args = parser.parse_args()

    args.est_out, args.gt_out = derive_output_paths(
        args.bag,
        args.est_out,
        args.gt_out,
    )

    est_rows = []
    gt_rows = []

    selected_topics = [
        args.velocity_topic,
        args.gt_velocity_topic,
    ]

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=selected_topics):
            t = stamp.to_sec()

            if topic == args.velocity_topic:
                est_rows.append(twist_message_to_row(t, msg))
            elif topic == args.gt_velocity_topic:
                gt_rows.append(twist_message_to_row(t, msg))

    df_est = build_velocity_dataframe(est_rows).sort_values("time")
    df_gt = build_velocity_dataframe(gt_rows).sort_values("time")

    df_est.to_csv(args.est_out, index=False)
    print(f"Saved estimate velocity CSV to {args.est_out}")
    print(f"Estimate rows: {len(df_est)}")

    df_gt.to_csv(args.gt_out, index=False)
    print(f"Saved GT velocity CSV to {args.gt_out}")
    print(f"GT rows: {len(df_gt)}")

# python3 1_rosbag_to_csv_vel.py --bag flight_6_27.bag 

if __name__ == "__main__":
    main()