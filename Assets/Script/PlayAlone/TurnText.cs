using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
public class TurnText : MonoBehaviour
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
        Debug.Log(turn.GetComponent<PlayAlone>());
        bool Turn = turn.GetComponent<PlayAlone>().isXTurn;
        if (Turn)
        {
            turnImage.sprite = xSprite.GetComponent<SpriteRenderer>().sprite;
        }
        else {
            turnImage.sprite = oSprite.GetComponent<SpriteRenderer>().sprite;
        }
    }
}
