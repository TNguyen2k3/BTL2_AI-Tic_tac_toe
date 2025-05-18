using System.Collections.Generic;
using UnityEngine;

public class BotLevel1
{
    public Vector2Int bestMove;
    private int[,] board;
    private int turn;
    private Vector2 startPosition;
    private float cellSize;
    private GameObject xSprite;
    private GameObject oSprite;
    private int maxDepth;
    private const int winLength = 5;
    private const int winScore = 1000000;
    private const int fourInRowScore = 100000;
    private const int threeInRowScore = 1000;
    private const int twoInRowScore = 100;

    public BotLevel1(int[,] board, int turn, Vector2 startPosition, float cellSize, GameObject xSprite, GameObject oSprite, int maxDepth)
    {
        this.board = board;
        this.turn = turn;
        this.startPosition = startPosition;
        this.cellSize = cellSize;
        this.xSprite = xSprite;
        this.oSprite = oSprite;
        this.maxDepth = Mathf.Max(1, maxDepth);
    }

    public void MakeMove()
    {
        List<Vector2Int> emptyCells = GetEmptyCells();
        if (emptyCells.Count == 0) return;

        // If center is available, take it first (good opening move)
        Vector2Int center = new Vector2Int(4, 4);
        if (board[4, 4] == 0 && emptyCells.Contains(center))
        {
            MakeMoveAt(center);
            return;
        }

        int bestScore = int.MinValue;
        bestMove = emptyCells[0];
        int sameScoreCount = 0;

        foreach (Vector2Int cell in emptyCells)
        {
            board[cell.x, cell.y] = turn;
            int score = Minimax(board, maxDepth - 1, int.MinValue, int.MaxValue, false);
            board[cell.x, cell.y] = 0;

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = cell;
                sameScoreCount = 1;
            }
            else if (score == bestScore)
            {
                sameScoreCount++;
                // Randomly select among moves with equal score to avoid predictability
                if (Random.Range(0, sameScoreCount) == 0)
                {
                    bestMove = cell;
                }
            }
        }

        MakeMoveAt(bestMove);
    }

    private void MakeMoveAt(Vector2Int cell)
    {
        Vector2 pos = new Vector2(
            startPosition.x + cell.x * cellSize,
            startPosition.y - cell.y * cellSize
        );
        GameObject piece = Object.Instantiate(
            turn == 1 ? xSprite : oSprite,
            pos,
            Quaternion.identity
        );
        board[cell.x, cell.y] = turn;
    }

    private int Minimax(int[,] boardState, int depth, int alpha, int beta, bool isMaximizing)
    {
        int evaluation = EvaluateBoard(boardState);
        if (Mathf.Abs(evaluation) >= winScore || depth == 0 || GetEmptyCells(boardState).Count == 0)
            return evaluation;

        if (isMaximizing)
        {
            int bestScore = int.MinValue;
            foreach (Vector2Int cell in GetEmptyCells(boardState))
            {
                boardState[cell.x, cell.y] = turn;
                int score = Minimax(boardState, depth - 1, alpha, beta, false);
                boardState[cell.x, cell.y] = 0;
                bestScore = Mathf.Max(score, bestScore);
                alpha = Mathf.Max(alpha, bestScore);
                if (beta <= alpha)
                    break;
            }
            return bestScore;
        }
        else
        {
            int bestScore = int.MaxValue;
            foreach (Vector2Int cell in GetEmptyCells(boardState))
            {
                boardState[cell.x, cell.y] = -turn;
                int score = Minimax(boardState, depth - 1, alpha, beta, true);
                boardState[cell.x, cell.y] = 0;
                bestScore = Mathf.Min(score, bestScore);
                beta = Mathf.Min(beta, bestScore);
                if (beta <= alpha)
                    break;
            }
            return bestScore;
        }
    }

    private int EvaluateBoard(int[,] boardState)
    {
        int score = 0;

        // Check all possible lines of 5
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (boardState[i, j] == 0) continue;

                int player = boardState[i, j];
                int playerMultiplier = (player == turn) ? 1 : -1;

                // Evaluate all four directions
                score += EvaluateLine(boardState, i, j, 1, 0) * playerMultiplier; // Horizontal
                score += EvaluateLine(boardState, i, j, 0, 1) * playerMultiplier; // Vertical
                score += EvaluateLine(boardState, i, j, 1, 1) * playerMultiplier; // Diagonal down-right
                score += EvaluateLine(boardState, i, j, 1, -1) * playerMultiplier; // Diagonal up-right
            }
        }

        // Add center control bonus
        if (boardState[4, 4] == turn) score += 10;
        else if (boardState[4, 4] == -turn) score -= 10;

        return score;
    }

    private int EvaluateLine(int[,] boardState, int x, int y, int dx, int dy)
    {
        int player = boardState[x, y];
        int score = 0;

        // Check all possible 5-in-a-row sequences including this cell
        for (int k = 0; k < winLength; k++)
        {
            int count = 0;
            int empty = 0;
            bool blocked = false;

            for (int l = 0; l < winLength; l++)
            {
                int nx = x + (k + l - winLength + 1) * dx;
                int ny = y + (k + l - winLength + 1) * dy;

                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9)
                {
                    blocked = true;
                    break;
                }

                if (boardState[nx, ny] == player) count++;
                else if (boardState[nx, ny] == 0) empty++;
                else
                {
                    blocked = true;
                    break;
                }
            }

            if (!blocked)
            {
                // Winning line
                if (count == winLength) return winScore;
                
                // Potential lines
                if (count == 4 && empty == 1) score += fourInRowScore;
                else if (count == 3 && empty == 2) score += threeInRowScore;
                else if (count == 2 && empty == 3) score += twoInRowScore;
            }
        }

        return score;
    }

    private List<Vector2Int> GetEmptyCells(int[,] boardState = null)
    {
        // Use the provided boardState or the default board
        var currentBoard = boardState ?? board;
        var result = new HashSet<Vector2Int>();
        int range = 2; 

        // --- Optimization 1: Handle Empty Board ---
        // Check if the board is empty. We still need to iterate, but can exit early.
        bool boardEmpty = true;
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (currentBoard[i, j] != 0)
                {
                    boardEmpty = false;
                    break; // Found a piece, board is not empty
                }
            }
            if (!boardEmpty) break; // Exit outer loop if board is not empty
        }

        if (boardEmpty)
        {
            // Return only the center position if the board is empty
            result.Add(new Vector2Int(4, 4)); 
            return new List<Vector2Int>(result);
        }

        // --- Optimization 2: Search around existing pieces efficiently ---
        // Iterate through the entire board to find pieces
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                // If the cell contains a piece
                if (currentBoard[i, j] != 0)
                {
                    // Check surrounding cells within the specified range
                    for (int di = -range; di <= range; di++)
                    {
                        for (int dj = -range; dj <= range; dj++)
                        {
                            // Calculate neighbor coordinates
                            int ni = i + di;
                            int nj = j + dj;

                            // Check if the neighbor is within board bounds (0-8)
                            if (ni >= 0 && ni < 9 && nj >= 0 && nj < 9)
                            {
                                // Check if the neighbor cell is empty
                                if (currentBoard[ni, nj] == 0)
                                {
                                    // Add the empty cell's coordinates to the result set.
                                    // HashSet automatically handles duplicates.
                                    result.Add(new Vector2Int(ni, nj));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert the HashSet to a List and return
        return new List<Vector2Int>(result);
    }

}