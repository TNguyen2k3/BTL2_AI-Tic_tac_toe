using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using TMPro;
using UnityEngine.SceneManagement;
using Unity.Barracuda;
[System.Serializable]
public class Trophy
{
    public string enemyName;
    public int win;
    public int lose;
    public int draw;

    public Trophy(string enemyName)
    {
        this.enemyName = enemyName;
        this.win = 0;
        this.lose = 0;
        this.draw = 0;
    }
}

[System.Serializable]
public class Bot
{
    public string botName;
    public List<Trophy> enemiesList = new List<Trophy>();

    public Bot(string name)
    {
        botName = name;
        enemiesList.Add(new Trophy("Level 0"));
        enemiesList.Add(new Trophy("Level 1"));
        enemiesList.Add(new Trophy("Level 2"));
        enemiesList.Add(new Trophy("Level 3"));
        enemiesList.Add(new Trophy("Level 4"));
    }
    

    public void UpdateResult(string enemyLevel, int isWin)
    {
        Trophy trophy = enemiesList.Find(t => t.enemyName == enemyLevel);
        if (trophy != null)
        {
            if (isWin == 1)
                trophy.win++;
            else if (isWin == -1)
                trophy.lose++;
            else trophy.draw++;
        }
    }
}
[System.Serializable]
public class BotListWrapper
{
    public List<Bot> bots = new List<Bot>();
}
public class BotVSBot : MonoBehaviour
{
    public NNModel dqnModelAsset;
    private DQNBot dqnBot;
    Bot Xbot;
    Bot Obot;

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
    private bool isLearned = false;
    private List<MoveLog> moveLogs = new List<MoveLog>();
    // Chuyển trạng thái bàn cờ hiện tại thành mảng float[81]
    float[] boardState = new float[81];

    void OnDestroy()
    {
        dqnBot?.Dispose();
    }
    // Start is called before the first frame update
    void Start()
    {
        Xbot = new Bot("Level " + (PlayerPrefs.GetInt("XBotLevel")).ToString());
        Obot = new Bot("Level " + (PlayerPrefs.GetInt("OBotLevel")).ToString());
        if (Xbot.botName == "Level 4" || Obot.botName == "Level 4")
        {
            isLearned = true;
            dqnBot = new DQNBot(dqnModelAsset);
        }
        PlayerPrefs.SetString("Scene", SceneName);
        InitializeBoard();

        int Role = PlayerPrefs.GetInt("Role");  // 0 = X, 1 = O, 2 = random
        float random = Random.Range(0f, 1f);  // tránh lỗi Random.Range(int, int)
        turn = (random < 0.5f) ? -1 : 1;
    }

    // Update is called once per frame
    void Update()
    {
        string Level;
        if (turn == 1) Level = Xbot.botName;
        else Level = Obot.botName;
        for (int x = 0; x < 9; x++)
        {
            for (int y = 0; y < 9; y++)
            {
                int index = x * 9 + y;
                if (board[x, y] == 1) boardState[index] = 1f;       // Bot
                else if (board[x, y] == -1) boardState[index] = -1f; // Người chơi
                else boardState[index] = 0f;                         // Ô trống
            }
        }
        
        if (Level == "Level 1")
        {
            IBotStrategy bot = new MinimaxBot((int)Random.Range(2, 3));
            Vector2Int move = bot.GetNextMove(board, turn);
            Vector2 piecePosition = new Vector2(startPosition.x + move.x * cellSize, startPosition.y - move.y * cellSize);
            Instantiate(turn == 1 ? xSprite : oSprite, piecePosition, Quaternion.identity);
            board[move.x, move.y] = turn;
            if (isLearned) LogMove(board, turn == 1 ? 1 : -1, move.x, move.y);
            turn = -turn;
        }
        else if (Level == "Level 3")
        {
            // int movesCount = CountMoves();
            // int adaptiveDepth = Mathf.Clamp(5 + movesCount / 10, 5, 7);
            BotLevel1 bot = new BotLevel1(board, turn, startPosition, cellSize, xSprite, oSprite, 3);
            bot.MakeMove();
            if (isLearned) LogMove(board, turn == 1 ? 1 : -1, bot.bestMove.x, bot.bestMove.y);
            turn = -turn;
        }
        else if (Level == "Level 2")
        {
            IBotStrategy bot = new AdvancedBot(400, 8); // 400ms max time, 8
            Vector2Int move = bot.GetNextMove(board, turn);
            Vector2 piecePosition = new Vector2(startPosition.x + move.x * cellSize, startPosition.y - move.y * cellSize);
            Instantiate(turn == 1 ? xSprite : oSprite, piecePosition, Quaternion.identity);
            board[move.x, move.y] = turn;
            if (isLearned) LogMove(board, turn == 1 ? 1 : -1, move.x, move.y);
            turn = -turn;
        }
        else if (Level == "Level 4")
        {
            if (!dqnModelAsset) BotDemo();
            else
            {
                Vector2Int move = dqnBot.PredictMove(boardState);
                Vector2 pieceWorldPosition = new Vector2(startPosition.x + move.x * cellSize, startPosition.y - move.y * cellSize);
                PlacePiece(pieceWorldPosition);
                Debug.Log(turn);
                
            }
        }
        else BotDemo();
        

        CheckWinCondition();
        if (result != 0)
        {
            PlayerPrefs.SetInt("Result", result);
            PlayerPrefs.Save(); // Lưu thay đổi ngay lập tức
            SceneManager.LoadScene("FinishedScene");
            if (result > 0) {
                Debug.Log("X is the Winner!");
                Xbot.UpdateResult(Obot.botName, 1);
                Obot.UpdateResult(Xbot.botName, -1);
            }

            else {
                Debug.Log("O is the Winner!");
                Xbot.UpdateResult(Obot.botName, -1);
                Obot.UpdateResult(Xbot.botName, 1);
            }

            SaveBotToFile(Xbot);
            SaveBotToFile(Obot);
        }
        else
        {
            CheckDraw();
            if (isFinished) {
                Xbot.UpdateResult(Obot.botName, 0);
                Obot.UpdateResult(Xbot.botName, 0);
                SaveBotToFile(Xbot);
                SaveBotToFile(Obot);
                Debug.Log("2 con bot hoa nhau");
                PlayerPrefs.SetInt("Result", result);
                PlayerPrefs.Save(); // Lưu thay đổi ngay lập tức
                SceneManager.LoadScene("FinishedScene");
            }
        }
        if (isFinished) SaveMatchLog();
    }

    void LogMove(int[,] boardState, int player, int x, int y)
    {
        int[,] boardCopy = (int[,])boardState.Clone(); // tránh tham chiếu
        MoveLog log = new MoveLog
        {
            board = boardCopy,
            player = player,
            x = x,
            y = y
        };
        moveLogs.Add(log);
    }
    void SaveMatchLog()
    {
        int botRole = (Xbot.botName == "Level 4") ? 1 : -1;
        int gameResult = result == 0 ? 0 : (result == botRole ? 1 : -1);
        string json = JsonUtility.ToJson(new Wrapper { moves = moveLogs.ToArray(), botRole = botRole, gameResult = gameResult}, true);
        string path = Application.dataPath + "/LearnData" + "/match_log_" + System.DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".json";
        System.IO.File.WriteAllText(path, json);
        Debug.Log("Saved match log to: " + path);
    }

    void SaveBotToFile(Bot bot)
    {
        string filePath = Application.dataPath + "/StreamingAssets/BotTrophy.json";

        // Nếu chưa có folder StreamingAssets, tạo folder
        if (!Directory.Exists(Path.GetDirectoryName(filePath)))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(filePath));
        }

        BotListWrapper botList = new BotListWrapper();

        if (File.Exists(filePath))
        {
            string existingJson = File.ReadAllText(filePath);
            botList = JsonUtility.FromJson<BotListWrapper>(existingJson);

            if (botList == null)
            {
                botList = new BotListWrapper(); // tránh lỗi null
            }
        }

        // Kiểm tra nếu đã tồn tại bot trùng tên
        Bot existingBot = botList.bots.Find(b => b.botName == bot.botName);
        if (existingBot != null)
        {
            // Cộng dồn win/lose theo từng Trophy
            foreach (var trophy in bot.enemiesList)
            {
                Trophy existingTrophy = existingBot.enemiesList.Find(t => t.enemyName == trophy.enemyName);
                if (existingTrophy != null)
                {
                    existingTrophy.win += trophy.win;
                    existingTrophy.lose += trophy.lose;
                    existingTrophy.draw += trophy.draw;
                }
                else
                {
                    // Nếu Trophy này chưa có thì thêm mới
                    existingBot.enemiesList.Add(new Trophy(trophy.enemyName) { win = trophy.win, lose = trophy.lose, draw = trophy.draw});
                }
            }
        }
        else
        {
            // Nếu bot này chưa có thì thêm mới
            botList.bots.Add(bot);
        }

        // Serialize lại toàn bộ
        string newJson = JsonUtility.ToJson(botList, true);
        File.WriteAllText(filePath, newJson);
        Debug.Log($"Đã lưu {bot.botName} vào {filePath}");
    }


    int CountMoves()
    {
        int count = 0;
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0)
                {
                    count++;
                }
            }
        }
        return count;
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
            Debug.Log(turn);
            if (isLearned) LogMove(board, turn == 1 ? 1 : -1, move.x, move.y);
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
            if (isLearned) LogMove(board, turn == 1 ? 1 : -1, xIndex, yIndex);
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
                    result = player; // Trả về người thắng (1 hoặc -1)
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
