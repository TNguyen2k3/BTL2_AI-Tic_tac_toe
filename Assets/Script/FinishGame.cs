using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
public class FinishGame : MonoBehaviour
{
    
    private int result;
    public TextMeshProUGUI winString;
    
    // Start is called before the first frame update
    void Start()
    {
        if (PlayerPrefs.HasKey("Result")){
            result = PlayerPrefs.GetInt("Result");
        }
        else return;
        if (result == 0) Debug.Log("Draw!");
        else if (result == 1) {
            winString.text = "X is the winner";
        }
        else winString.text = "O is the winner";

    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
