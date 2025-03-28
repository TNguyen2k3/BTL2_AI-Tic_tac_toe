using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class TurnImage : MonoBehaviour
{
     [SerializeField] Image turnImage;
    GameObject turn;
    public GameObject xSprite;
    public GameObject oSprite;
    // Start is called before the first frame update
    void Start()
    {
        turn = GameObject.FindGameObjectWithTag("MainCamera");
    }

    // Update is called once per frame
    void Update()
    {
        
        int Turn = turn.GetComponent<PlayWithBot>().turn;
        if (Turn == 1)
        {
            turnImage.sprite = xSprite.GetComponent<SpriteRenderer>().sprite;
        }
        else {
            turnImage.sprite = oSprite.GetComponent<SpriteRenderer>().sprite;
        }
    }
}
