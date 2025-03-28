using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
public class ButtonChooseLevelHandling : MonoBehaviour
{
    // Start is called before the first frame update
    public Button ChooseLevelButton;  
    string SceneName;
    void Start()
    {
        if(PlayerPrefs.HasKey("Scene")) SceneName = PlayerPrefs.GetString("Scene");
    }

    // Update is called once per frame
    void Update()
    {
        if (SceneName == null) return;
        if (SceneName == "Play alone") {
            ChooseLevelButton.gameObject.SetActive(false);
        }
    }
}
