using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.SceneManagement;
public class Go_to_a_scene : MonoBehaviour
{
    public string sceneName;
    // Start is called before the first frame update
    public void OnButtonClick(){
        SceneManager.LoadScene(sceneName);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
