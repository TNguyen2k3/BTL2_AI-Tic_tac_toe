using UnityEditor;
using UnityEngine;

namespace Common.Editor
{
    [InitializeOnLoad]
    public class ResetScene
    {
        static ResetScene()
        {
            SceneView.duringSceneGui -= DrawEditor;
            SceneView.duringSceneGui += DrawEditor;
        }

        private static void DrawEditor(SceneView sceneView)
        {
            Rect sceneRect = sceneView.position;

            Handles.BeginGUI();
            GUILayout.BeginArea(new Rect(6, sceneRect.height - 50, sceneRect.width - 12, 19));
            GUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();

            ResetCamera();

            GUILayout.EndHorizontal();
            GUILayout.EndArea();
            Handles.EndGUI();
        }

        [MenuItem("GameObject/− Reset Camera Top-Down Ortho", false, -1)]
        private static void ResetCameraMenu()
        {
            ResetCameraTopDownOrtho();
        }

        private static void ResetCamera()
        {
            GUIContent content = EditorGUIUtility.IconContent("SceneViewCamera");
            content.text = " Reset Camera";
            content.tooltip = "Set the camera's to the game's angle";
            if (!GUILayout.Button(content))
                return;

            ResetCameraTopDownOrtho();
        }

        private static void ResetCameraTopDownOrtho()
        {
            SceneView view = SceneView.lastActiveSceneView;
            if (view == null)
                return;

            view.rotation = Quaternion.Euler(45, 0, 0);
            view.orthographic = true;

        }

    }
}