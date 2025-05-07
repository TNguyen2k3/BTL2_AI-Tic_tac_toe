using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
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
        if (SceneName == "PlayAlone") {
            ChooseLevelButton.gameObject.SetActive(false);
        }
    }
    public void OnButtonClicked(){
        if (SceneName == "PlayWithBot"){
            SceneManager.LoadScene("ChooseLevel");
        }
        else if (SceneName == "BotVSBot"){
            SceneManager.LoadScene("ChooseBots");
        }
    }
}
