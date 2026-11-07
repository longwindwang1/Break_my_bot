using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Velocity command (vx, vy, wz) in the Isaac base frame: x forward, y left, wz > 0 turns left.
    /// Fixed (for trials) or from keyboard / gamepad, clamped to the training ranges in robot_interface.yaml
    /// `commands`. Keys: W/S forward/back, A/D left/right, Q/E turn left/right, Tab toggles fixed/manual,
    /// R resets the robot. Gamepad: left stick translate, right stick turn, South button resets.
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

        void Awake() => ranges = RobotInterface.Load().Commands;

        void Update()
        {
            var keyboard = Keyboard.current;
            var gamepad = Gamepad.current;
            if (keyboard != null && keyboard.tabKey.wasPressedThisFrame)
                useFixedCommand = !useFixedCommand;
            if ((keyboard != null && keyboard.rKey.wasPressedThisFrame) || (gamepad != null && gamepad.buttonSouth.wasPressedThisFrame))
                controller?.ResetRobot();

            var raw = useFixedCommand ? fixedCommand : Manual(keyboard, gamepad);
            Command = new Vector3(
                Mathf.Clamp(raw.x, ranges.LinVelX[0], ranges.LinVelX[1]),
                Mathf.Clamp(raw.y, ranges.LinVelY[0], ranges.LinVelY[1]),
                Mathf.Clamp(raw.z, ranges.AngVelZ[0], ranges.AngVelZ[1]));
        }

        Vector3 Manual(Keyboard keyboard, Gamepad gamepad)
        {
            float forward = 0f, left = 0f, turnLeft = 0f;
            if (gamepad != null)
            {
                forward = gamepad.leftStick.y.ReadValue();
                left = -gamepad.leftStick.x.ReadValue();
                turnLeft = -gamepad.rightStick.x.ReadValue();
            }
            if (keyboard != null)
            {
                forward += Key(keyboard.wKey) - Key(keyboard.sKey);
                left += Key(keyboard.aKey) - Key(keyboard.dKey);
                turnLeft += Key(keyboard.qKey) - Key(keyboard.eKey);
            }
            forward = Mathf.Clamp(forward, -1f, 1f) * manualScale;
            left = Mathf.Clamp(left, -1f, 1f) * manualScale;
            turnLeft = Mathf.Clamp(turnLeft, -1f, 1f) * manualScale;
            return new Vector3(
                forward * (forward >= 0 ? ranges.LinVelX[1] : -ranges.LinVelX[0]),
                left * (left >= 0 ? ranges.LinVelY[1] : -ranges.LinVelY[0]),
                turnLeft * (turnLeft >= 0 ? ranges.AngVelZ[1] : -ranges.AngVelZ[0]));
        }

        static float Key(UnityEngine.InputSystem.Controls.KeyControl key) => key.isPressed ? 1f : 0f;
    }
}
