using UnityEditor;
using UnityEngine;
using System.IO;

public static class CopyAssetPath
{
    [MenuItem("Assets/Copy Full Path", false, 2000)]
    private static void CopyPath()
    {
        // Get the selected asset path
        string path = AssetDatabase.GetAssetPath(Selection.activeObject);
        string fullPath = Path.GetFullPath(path);

        // Copy the path to the system clipboard
        GUIUtility.systemCopyBuffer = fullPath;

    }

}
