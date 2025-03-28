using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class PlayAgain : MonoBehaviour
{
    // Start is called before the first frame update
    string SceneName;
    void Start()
    {
        if(PlayerPrefs.HasKey("Scene")) SceneName = PlayerPrefs.GetString("Scene");
    }
    public void OnButtonClick(){
        if (SceneName != null) SceneManager.LoadScene(SceneName);  // Load the last saved scene
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
