using UnityEngine;

namespace BMB
{
    /// <summary>Third-person camera that trails a target (the Go2 base) and follows its heading.</summary>
    public sealed class FollowCamera : MonoBehaviour
    {
        [SerializeField] Transform target;
        [Tooltip("Offset in the target's yaw frame (Unity axes: z forward, y up)")]
        [SerializeField] Vector3 offset = new Vector3(0.9f, 0.6f, -1.4f);
        [SerializeField] float positionLag = 0.15f;

        public Transform Target { get => target; set => target = value; }

        void LateUpdate()
        {
            if (target == null)
                return;
            var yaw = Quaternion.Euler(0f, target.eulerAngles.y, 0f);
            var desired = target.position + yaw * offset;
            var t = positionLag <= 0f ? 1f : 1f - Mathf.Exp(-Time.deltaTime / positionLag);
            transform.position = Vector3.Lerp(transform.position, desired, t);
            transform.LookAt(target.position + Vector3.up * 0.05f);
        }
    }
}
