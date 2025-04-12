using System.Collections.Generic;
using UnityEngine;

public class BotLevel1
{
    private int[,] board;
    private int turn;

    private Vector2 startPosition;
    private float cellSize;
    private GameObject xSprite;
    private GameObject oSprite;
    private int maxDepth;

    public BotLevel1(int[,] board, int turn, Vector2 startPosition, float cellSize, GameObject xSprite, GameObject oSprite, int maxDepth = 3)
    {
        this.board = board;
        this.turn = turn;
        this.startPosition = startPosition;
        this.cellSize = cellSize;
        this.xSprite = xSprite;
        this.oSprite = oSprite;
        this.maxDepth = maxDepth;
    }

    public void MakeMove()
    {
        int bestScore = int.MinValue;
        Vector2Int bestMove = new Vector2Int(-1, -1);

        foreach (Vector2Int cell in GetEmptyCells())
        {
            board[cell.x, cell.y] = turn;
            int score = Minimax(board, false, maxDepth);
            board[cell.x, cell.y] = 0;

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = cell;
            }
        }

        if (bestMove.x != -1)
        {
            Vector2 pos = new Vector2(startPosition.x + bestMove.x * cellSize, startPosition.y - bestMove.y * cellSize);
            GameObject piece = Object.Instantiate(turn == 1 ? xSprite : oSprite, pos, Quaternion.identity);
            board[bestMove.x, bestMove.y] = turn;
        }
    }

    private int Minimax(int[,] boardState, bool isMaximizing, int depth)
    {
        int evaluation = EvaluateBoard();
        if (Mathf.Abs(evaluation) == 1000 || depth == 0 || GetEmptyCells().Count == 0)
            return evaluation;

        if (isMaximizing)
        {
            int bestScore = int.MinValue;
            foreach (Vector2Int cell in GetEmptyCells())
            {
                boardState[cell.x, cell.y] = turn;
                int score = Minimax(boardState, false, depth - 1);
                boardState[cell.x, cell.y] = 0;
                bestScore = Mathf.Max(score, bestScore);
            }
            return bestScore;
        }
        else
        {
            int bestScore = int.MaxValue;
            foreach (Vector2Int cell in GetEmptyCells())
            {
                boardState[cell.x, cell.y] = -turn;
                int score = Minimax(boardState, true, depth - 1);
                boardState[cell.x, cell.y] = 0;
                bestScore = Mathf.Min(score, bestScore);
            }
            return bestScore;
        }
    }

    private int EvaluateBoard()
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                int p = board[i, j];
                if (p != 0 && (
                    CheckDirection(i, j, 1, 0, p, 5) ||
                    CheckDirection(i, j, 0, 1, p, 5) ||
                    CheckDirection(i, j, 1, 1, p, 5) ||
                    CheckDirection(i, j, 1, -1, p, 5)))
                {
                    return p * 1000;
                }
            }
        }
        return 0;
    }

    private bool CheckDirection(int x, int y, int dx, int dy, int player, int winLength)
    {
        for (int k = 0; k < winLength; k++)
        {
            int nx = x + k * dx;
            int ny = y + k * dy;
            if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9)
                return false;
            if (board[nx, ny] != player)
                return false;
        }
        return true;
    }

    // Get empty cells bằng cách giới hạn phạm vi gần các quân đã đánh
    private List<Vector2Int> GetEmptyCells()
    {
        HashSet<Vector2Int> result = new HashSet<Vector2Int>();
        int range = 1; // Giới hạn phạm vi ô trống xung quanh các quân đã đánh

        // Duyệt qua tất cả các ô trên bàn cờ
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0) // Nếu ô đã có quân
                {
                    // Kiểm tra các ô trống xung quanh
                    for (int dx = -range; dx <= range; dx++)
                    {
                        for (int dy = -range; dy <= range; dy++)
                        {
                            int ni = i + dx;
                            int nj = j + dy;
                            // Kiểm tra xem ô mới có hợp lệ và trống hay không
                            if (ni >= 0 && ni < 9 && nj >= 0 && nj < 9 && board[ni, nj] == 0)
                            {
                                result.Add(new Vector2Int(ni, nj));
                            }
                        }
                    }
                }
            }
        }

        return new List<Vector2Int>(result); // Trả về danh sách các ô trống đã được giới hạn
    }
}
