using UnityEngine;

namespace BMB
{
    /// <summary>
    /// A circle drawn on the ground under the mouse (tool footprint). A plain scene object (no DontSave), so it goes
    /// away with the play-mode scene instead of lingering in the editor.
    /// </summary>
    public sealed class BrushCursor
    {
        const int Segments = 48;
        readonly LineRenderer line;

        public BrushCursor(Color colour)
        {
            var go = new GameObject("BrushCursor");
            line = go.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.loop = true;
            line.positionCount = Segments;
            line.widthMultiplier = 0.02f;
            line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            line.sharedMaterial = new Material(Shader.Find("Sprites/Default"));
            line.startColor = line.endColor = colour;
            go.SetActive(false);
        }

        public void Show(Vector3 centre, float radius, Color colour)
        {
            line.gameObject.SetActive(true);
            line.startColor = line.endColor = colour;
            for (var i = 0; i < Segments; i++)
            {
                var a = i * Mathf.PI * 2f / Segments;
                line.SetPosition(i, centre + new Vector3(Mathf.Cos(a) * radius, 0.02f, Mathf.Sin(a) * radius));
            }
        }

        /// <summary>Outline of an axis-aligned (in yaw) rectangle on the ground.</summary>
        public void ShowRect(Vector3 centre, Vector2 size, float yawDeg, Color colour)
        {
            line.gameObject.SetActive(true);
            line.startColor = line.endColor = colour;
            var r = Quaternion.Euler(0f, yawDeg, 0f);
            var corners = new[] { new Vector3(-1, 0, -1), new Vector3(-1, 0, 1), new Vector3(1, 0, 1), new Vector3(1, 0, -1) };
            for (var i = 0; i < Segments; i++)
            {
                var t = i / (float)Segments * 4f;
                var k = Mathf.FloorToInt(t);
                var p = Vector3.Lerp(corners[k % 4], corners[(k + 1) % 4], t - k);
                line.SetPosition(i, centre + r * new Vector3(p.x * size.x / 2f, 0.02f, p.z * size.y / 2f));
            }
        }

        public void Hide() => line.gameObject.SetActive(false);

        public void Destroy()
        {
            if (line == null)
                return;
            Object.Destroy(line.sharedMaterial);
            Object.Destroy(line.gameObject);
        }
    }
}
