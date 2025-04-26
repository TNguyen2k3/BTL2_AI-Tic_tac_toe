using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class BotvsbotSelection : MonoBehaviour
{
    public TMP_Dropdown Bot;
    public string Key;
    
    // Start is called before the first frame update
    void Start()
    {
        int BotValue = Bot.value; 
        
        // Debug.Log( Key + " option: " + selectedOption);
        PlayerPrefs.SetInt(Key, BotValue);
        
        PlayerPrefs.Save();
    }

    public void OnSelectionChanged(){
        int BotValue = Bot.value; 
        
        // Debug.Log( Key + " option: " + selectedOption);
        PlayerPrefs.SetInt(Key, BotValue);
        
        PlayerPrefs.Save();
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
