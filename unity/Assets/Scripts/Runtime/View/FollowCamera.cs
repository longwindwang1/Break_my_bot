using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Third-person camera that trails a target (the Go2 base) and follows its heading. Hold the right mouse button and
    /// drag to orbit around the robot; Ctrl + mouse wheel zooms (the plain wheel belongs to the disturbance tools).
    /// </summary>
    public sealed class FollowCamera : MonoBehaviour
    {
        [SerializeField] Transform target;
        [Tooltip("Offset in the target's yaw frame (Unity axes: z forward, y up)")]
        [SerializeField] Vector3 offset = new Vector3(0.9f, 0.6f, -1.4f);
        [SerializeField] float positionLag = 0.15f;
        [SerializeField] float orbitSpeed = 0.25f;  // degrees per pixel

        float orbitYaw, orbitPitch, zoom = 1f;

        public Transform Target { get => target; set => target = value; }

        void LateUpdate()
        {
            if (target == null)
                return;
            ReadOrbitInput();
            var yaw = Quaternion.Euler(0f, target.eulerAngles.y + orbitYaw, 0f);
            var pitched = Quaternion.AngleAxis(orbitPitch, Vector3.right) * offset * zoom;
            var desired = target.position + yaw * pitched;
            var t = positionLag <= 0f ? 1f : 1f - Mathf.Exp(-Time.deltaTime / positionLag);
            transform.position = Vector3.Lerp(transform.position, desired, t);
            transform.LookAt(target.position + Vector3.up * 0.05f);
        }

        void ReadOrbitInput()
        {
            var mouse = Mouse.current;
            if (mouse == null)
                return;
            if (mouse.rightButton.isPressed)
            {
                var d = mouse.delta.ReadValue();
                orbitYaw += d.x * orbitSpeed;
                orbitPitch = Mathf.Clamp(orbitPitch - d.y * orbitSpeed, -25f, 60f);
            }
            var keyboard = Keyboard.current;
            if (keyboard != null && keyboard.ctrlKey.isPressed)
            {
                var s = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(s) > 0.01f)
                    zoom = Mathf.Clamp(zoom * (s > 0 ? 0.9f : 1.1f), 0.4f, 4f);
            }
        }
    }
}
