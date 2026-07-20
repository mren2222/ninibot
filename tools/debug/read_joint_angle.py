import time

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


PORT = "/dev/ttyACM0"


def main():
    print("Connecting to SO101 / SOFollower...")
    print(f"Port: {PORT}")

    config = SOFollowerRobotConfig(port=PORT)
    robot = SOFollower(config)

    robot.connect(calibrate=False)

    print("Connected.")

    print("Disabling torque...")
    robot.bus.disable_torque()
    print("Torque disabled. You can gently move the arm by hand now.")

    print("Reading joint observations...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            obs = robot.get_observation()

            print("Current observation:")
            for key, value in obs.items():
                print(f"  {key}: {value}")

            print("-" * 40)
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()