using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;
public class PlayWithBot : MonoBehaviour
{
    [SerializeField] TMP_Text level;

    public GameObject xSprite;
    public GameObject oSprite;
    // 1 is X, 2 is O, 0 is nothing
    private int[,] board = new int[9, 9];
    public int turn;
    private Vector2 startPosition = new Vector2(-16, 0);
    private float cellSize = 2f;
    public int result = 0;
    public bool isFinished = false;
    public int player;
    public string SceneName;
    // Start is called before the first frame update
    void Start()
    {
        PlayerPrefs.SetString("Scene", SceneName);
        InitializeBoard();
        int Role = PlayerPrefs.GetInt("Role");
        
        
        level.text = "Level " + level;
        if (Role == 1) player = -1; // O
        else if (Role == 0) player = 1; // X
        else {
            float random = Random.Range(0, 1);
            if (random < 0.5) player = -1; // O
            else player = 1; // X
        }
        float randomTurn = Random.Range(0f, 1f);
        Debug.Log(randomTurn);
        if (randomTurn < 0.5) turn = player;
        else turn = -player;

    }
    // Update is called once per frame
    void Update()
    {
        int Level = PlayerPrefs.GetInt("Level") + 1;
        if (turn != player) {
            if (Level == 1) {
                BotLevel1 bot = new BotLevel1(board, turn, startPosition, cellSize, xSprite, oSprite, 5);
                bot.MakeMove();
                turn = -turn;
            }
            else if (Level == 2) BotDemo();
            else BotDemo();
        }
        
        if (Input.GetMouseButtonDown(0))
        {            
            Vector2 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            if (turn == player) PlacePiece(mousePosition);
        }
        CheckWinCondition();
            
        if (result != 0) {
            PlayerPrefs.SetInt("Result", result);
            PlayerPrefs.Save(); // Lưu thay đổi ngay lập tức
            SceneManager.LoadScene("FinishedScene");
        }
        else{
            CheckDraw();
            if (isFinished) {
                PlayerPrefs.SetInt("Result", result);
                PlayerPrefs.Save(); // Lưu thay đổi ngay lập tức
                SceneManager.LoadScene("FinishedScene");
            }
        }
    }

    void BotDemo()
    {
        List<Vector2Int> emptyCells = new List<Vector2Int>();

        // Duyệt qua toàn bộ bàn cờ để tìm ô trống
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] == 0) 
                {
                    emptyCells.Add(new Vector2Int(i, j));
                }
            }
        }

        if (emptyCells.Count > 0)
        {
            // Chọn một ô trống ngẫu nhiên
            Vector2Int move = emptyCells[Random.Range(0, emptyCells.Count)];
            
            // Chuyển đổi vị trí sang tọa độ trong game
            Vector2 piecePosition = new Vector2(startPosition.x + move.x * cellSize, startPosition.y - move.y * cellSize);
            
            // Đặt quân cờ
            GameObject piece = Instantiate(turn == 1 ? xSprite : oSprite, piecePosition, Quaternion.identity);
            board[move.x, move.y] = (turn == 1 ? 1 : -1);

            // Đổi lượt
            turn = -turn;
        }
    }
    void PlacePiece(Vector2 position)
    {
        int xIndex = Mathf.RoundToInt((position.x - startPosition.x) / cellSize);
        int yIndex = Mathf.RoundToInt((startPosition.y - position.y) / cellSize);
        if (xIndex >= 0 && xIndex < 9 && yIndex >= 0 && yIndex < 9 && board[xIndex, yIndex] == 0)
        {
            GameObject piece = Instantiate(turn == 1 ? xSprite : oSprite, 
                new Vector2(startPosition.x + xIndex * cellSize, startPosition.y - yIndex * cellSize), 
                Quaternion.identity);
            board[xIndex, yIndex] = (turn == 1 ? 1 : -1);
            turn = -turn;
        }
    }
    void InitializeBoard()
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                board[i, j] = 0;
            }
        }
    }
    void CheckDraw(){
        for (int i = 0; i < 9; i++){
            for (int j = 0; j < 9; j++){
                
                if (board[i, j] == 0) return;
            }
        }
        result = 0;
        isFinished = true; // Hết bàn cờ, đánh dấu trạng thái kết thúc
    }
    // Check if there is a winner
    // return 1 if X win, -1 if O win
    void CheckWinCondition()
    {
        int winLength = 5; // Số quân cần để thắng (Caro thường là 5)
        
        // Duyệt qua toàn bộ bàn cờ
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                
                int player = board[i, j]; // Lấy giá trị ô hiện tại (1 = X, 2 = O)

                if (player == 0) continue; // Nếu ô trống, bỏ qua

                // Kiểm tra theo 4 hướng
                if (CheckDirection(i, j, 1, 0, player, winLength) ||  // Ngang →
                    CheckDirection(i, j, 0, 1, player, winLength) ||  // Dọc ↓
                    CheckDirection(i, j, 1, 1, player, winLength) ||  // Chéo chính ↘
                    CheckDirection(i, j, 1, -1, player, winLength))   // Chéo phụ ↙
                {
                    isFinished = true; // Thắng rồi, đánh dấu trạng thái kết thúc
                    result = player; // Trả về người thắng (1 hoặc 2)
                    return;
                }
            }
        }

        result = 0; // Không có ai thắng
    }

    // Hàm kiểm tra theo hướng nhất định
    bool CheckDirection(int x, int y, int dx, int dy, int player, int winLength)
    {
        int count = 0;
        
        for (int k = 0; k < winLength; k++)
        {
            int nx = x + k * dx;
            int ny = y + k * dy;

            // Kiểm tra nếu ra ngoài biên
            if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9)
                return false;

            // Kiểm tra nếu không phải quân của người chơi đang xét
            if (board[nx, ny] != player)
                return false;

            count++;
            // Debug.Log(count);
        }
       
        return count == winLength;
    }
}
