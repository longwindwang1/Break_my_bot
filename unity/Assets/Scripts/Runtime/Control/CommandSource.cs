using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Velocity command (vx, vy, wz) in the Isaac base frame: x forward, y left, wz > 0 turns left.
    /// Fixed (for trials) or manual from keyboard / gamepad / the on-screen stick and turn buttons, clamped to the
    /// training ranges in robot_interface.yaml `commands`. Keys: W/S forward/back, A/D left/right, Q/E turn left/right,
    /// Tab toggles fixed/manual, R resets the robot. Gamepad: left stick translate, right stick turn, South resets.
    /// </summary>
    public sealed class CommandSource : MonoBehaviour
    {
        [SerializeField] bool useFixedCommand = true;
        [Tooltip("Isaac (vx m/s, vy m/s, wz rad/s); the reference rollouts use (0.5, 0, 0)")]
        [SerializeField] Vector3 fixedCommand = new Vector3(0.5f, 0f, 0f);
        [Tooltip("Fraction of the command range used by keys / full stick deflection")]
        [Range(0f, 1f)] [SerializeField] float manualScale = 1f;
        [SerializeField] Go2Controller controller;

        CommandsSpec ranges;

        public Vector3 Command { get; private set; }
        public bool UseFixedCommand => useFixedCommand;

        /// <summary>On-screen stick, each axis in [−1, 1]: x = right, y = forward.</summary>
        public Vector2 UiStick { get; set; }
        /// <summary>On-screen turn buttons in [−1, 1]: +1 = turn left.</summary>
        public float UiTurn { get; set; }

        /// <summary>Forward speed of the fixed command (m/s).</summary>
        public float FixedSpeed
        {
            get => fixedCommand.x;
            set => fixedCommand.x = value;
        }

        public CommandsSpec Ranges => ranges ??= RobotInterface.Load().Commands;

        public void SetFixed(bool useFixed) => useFixedCommand = useFixed;

        void Awake() => ranges ??= RobotInterface.Load().Commands;

        void Update()
        {
            var keyboard = Keyboard.current;
            var gamepad = Gamepad.current;
            if (keyboard != null && keyboard.tabKey.wasPressedThisFrame)
                useFixedCommand = !useFixedCommand;
            if ((keyboard != null && keyboard.rKey.wasPressedThisFrame) || (gamepad != null && gamepad.buttonSouth.wasPressedThisFrame))
                controller?.ResetRobot();
            Command = Compute(keyboard, gamepad);
        }

        /// <summary>The command from the current inputs (fixed, or keyboard + gamepad + on-screen controls), clamped.</summary>
        public Vector3 Compute(Keyboard keyboard, Gamepad gamepad)
        {
            var r = Ranges;
            var raw = useFixedCommand ? fixedCommand : Manual(keyboard, gamepad);
            return new Vector3(
                Mathf.Clamp(raw.x, r.LinVelX[0], r.LinVelX[1]),
                Mathf.Clamp(raw.y, r.LinVelY[0], r.LinVelY[1]),
                Mathf.Clamp(raw.z, r.AngVelZ[0], r.AngVelZ[1]));
        }

        Vector3 Manual(Keyboard keyboard, Gamepad gamepad)
        {
            float forward = UiStick.y, left = -UiStick.x, turnLeft = UiTurn;
            if (gamepad != null)
            {
                forward += gamepad.leftStick.y.ReadValue();
                left -= gamepad.leftStick.x.ReadValue();
                turnLeft -= gamepad.rightStick.x.ReadValue();
            }
            if (keyboard != null)
            {
                forward += Key(keyboard.wKey) - Key(keyboard.sKey);
                left += Key(keyboard.aKey) - Key(keyboard.dKey);
                turnLeft += Key(keyboard.qKey) - Key(keyboard.eKey);
            }
            var r = Ranges;
            forward = Mathf.Clamp(forward, -1f, 1f) * manualScale;
            left = Mathf.Clamp(left, -1f, 1f) * manualScale;
            turnLeft = Mathf.Clamp(turnLeft, -1f, 1f) * manualScale;
            return new Vector3(
                forward * (forward >= 0 ? r.LinVelX[1] : -r.LinVelX[0]),
                left * (left >= 0 ? r.LinVelY[1] : -r.LinVelY[0]),
                turnLeft * (turnLeft >= 0 ? r.AngVelZ[1] : -r.AngVelZ[0]));
        }

        static float Key(UnityEngine.InputSystem.Controls.KeyControl key) => key.isPressed ? 1f : 0f;
    }
}
