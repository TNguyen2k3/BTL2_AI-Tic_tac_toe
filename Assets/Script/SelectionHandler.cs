using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
public class SelectionHandler : MonoBehaviour
{
    public TMP_Dropdown dropdown;
    public string Key;
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log(dropdown.value.GetType());
        int selectedOption = dropdown.value; // 0 = X, 1 = O, 2 = Random
        Debug.Log( Key + " option: " + selectedOption);
        PlayerPrefs.SetInt(Key, selectedOption);
        PlayerPrefs.Save();
    }
    public void OnSelectionChanged()
    {
        // bug
        int selectedOption = dropdown.value; // 0 = X, 1 = O, 2 = Random
        Debug.Log( Key + " option: " + selectedOption);
        PlayerPrefs.SetInt(Key, selectedOption);
        PlayerPrefs.Save();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
