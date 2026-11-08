using UnityEngine;
using UnityEngine.TextCore.Text;
using UnityEngine.UIElements;

namespace BMB
{
    /// <summary>
    /// Fonts for the game UI, created at runtime from fonts installed on the machine (no font files in the project):
    /// Bahnschrift for text (a DIN-like face that suits the instrument look), Consolas for numbers, each falling back
    /// to Segoe UI for missing glyphs (Greek μ, Δ, symbols). If none is installed the default UI font stays in place.
    /// </summary>
    public static class UiFonts
    {
        static bool loaded;
        static FontAsset text, mono;

        public static FontAsset Text { get { Load(); return text; } }
        public static FontAsset Mono { get { Load(); return mono; } }

        static void Load()
        {
            if (loaded)
                return;
            loaded = true;
            var fallback = Create("Segoe UI", "Regular");
            text = Create("Bahnschrift", "Regular") ?? fallback;
            mono = Create("Consolas", "Regular") ?? Create("Cascadia Mono", "Regular") ?? text;
            foreach (var f in new[] { text, mono })
                if (f != null && fallback != null && f != fallback)
                {
                    f.fallbackFontAssetTable ??= new System.Collections.Generic.List<FontAsset>();
                    if (!f.fallbackFontAssetTable.Contains(fallback))
                        f.fallbackFontAssetTable.Add(fallback);
                }
        }

        static FontAsset Create(string family, string style)
        {
            try
            {
                var f = FontAsset.CreateFontAsset(family, style);
                if (f != null)
                    f.hideFlags = HideFlags.DontSave;
                return f;
            }
            catch (System.Exception)
            {
                return null;
            }
        }

        /// <summary>Text font on a subtree (call on the root); numbers (class "num") get the monospace font.</summary>
        public static void Apply(VisualElement root)
        {
            if (Text != null)
                root.style.unityFontDefinition = FontDefinition.FromSDFFont(Text);
            ApplyMono(root);
        }

        /// <summary>Monospace font on every element with class "num" under (and including) the element.</summary>
        public static void ApplyMono(VisualElement element)
        {
            if (Mono == null)
                return;
            element.Query(className: "num").ForEach(e => e.style.unityFontDefinition = FontDefinition.FromSDFFont(Mono));
            if (element.ClassListContains("num"))
                element.style.unityFontDefinition = FontDefinition.FromSDFFont(Mono);
        }
    }
}
