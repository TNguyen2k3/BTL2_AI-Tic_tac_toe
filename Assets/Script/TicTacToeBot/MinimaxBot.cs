using System.Collections.Generic;
using UnityEngine;

public class MinimaxBot : IBotStrategy
{
    private int maxDepth;

    public MinimaxBot(int depth = 2)
    {
        this.maxDepth = Mathf.Clamp(depth, 1, 4);
    }

    public Vector2Int GetNextMove(int[,] board, int turn)
    {
        List<Vector2Int> moves = GetCandidateMoves(board);
        int bestScore = int.MinValue;
        Vector2Int bestMove = moves[0];

        foreach (var move in moves)
        {
            board[move.x, move.y] = turn; //Giả lập quân -> Đánh dấu ô này đã được đi quân X hay O (1 hay -1)
            int score = Minimax(board, maxDepth, false, turn);
            board[move.x, move.y] = 0; //Bỏ đánh dấu

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = move;
            }
        }

        return bestMove;
    }

    private int Minimax(int[,] board, int depth, bool isMaximizing, int player)
    {
        int winner = CheckWin(board);
        if (winner == player) return 100000;
        if (winner == -player) return -100000;
        if (depth == 0 || IsFull(board)) return Evaluate(board, player);

        int best = isMaximizing ? int.MinValue : int.MaxValue;
        var moves = GetCandidateMoves(board);

        foreach (var move in moves)
        {
            board[move.x, move.y] = isMaximizing ? player : -player;
            int score = Minimax(board, depth - 1, !isMaximizing, player);
            board[move.x, move.y] = 0;

            if (isMaximizing)
                best = Mathf.Max(best, score);
            else
                best = Mathf.Min(best, score);
        }

        return best;
    }

    private int Evaluate(int[,] board, int player)
    {
        return EvaluatePlayer(board, player) - EvaluatePlayer(board, -player);
    }

    private int EvaluatePlayer(int[,] board, int who)
    {
        int score = 0;
        int[][] directions = new int[][] {
            new int[] {1, 0}, new int[] {0, 1},
            new int[] {1, 1}, new int[] {1, -1}
        };

        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != who) continue;

                foreach (var dir in directions)
                {
                    int count = 1;
                    int openEnds = 0;

                    int ni = i - dir[0], nj = j - dir[1];
                    if (InBounds(ni, nj) && board[ni, nj] == 0) openEnds++;

                    for (int k = 1; k < 5; k++)
                    {
                        ni = i + dir[0] * k;
                        nj = j + dir[1] * k;

                        if (!InBounds(ni, nj)) break;
                        if (board[ni, nj] == who) count++;
                        else if (board[ni, nj] == 0) { openEnds++; break; }
                        else break;
                    }

                    if (count >= 5) score += 100000;
                    else if (count == 4 && openEnds == 2) score += 10000;
                    else if (count == 3 && openEnds == 2) score += 1000;
                    else if (count == 2 && openEnds == 2) score += 100;
                }
            }
        }

        return score;
    }

    private bool InBounds(int x, int y)
    {
        return x >= 0 && x < 9 && y >= 0 && y < 9;
    }

    private int CheckWin(int[,] board)
    {
        int[][] directions = new int[][] {
            new int[] {1, 0}, new int[] {0, 1},
            new int[] {1, 1}, new int[] {1, -1}
        };

        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                int player = board[i, j];
                if (player == 0) continue;

                foreach (var dir in directions)
                {
                    int count = 1;
                    for (int k = 1; k < 5; k++)
                    {
                        int ni = i + dir[0] * k, nj = j + dir[1] * k;
                        if (!InBounds(ni, nj) || board[ni, nj] != player) break;
                        count++;
                    }
                    if (count >= 5) return player;
                }
            }
        }
        return 0;
    }

    private bool IsFull(int[,] board)
    {
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (board[i, j] == 0) return false;
        return true;
    }

    private List<Vector2Int> GetCandidateMoves(int[,] board)
    {
        HashSet<Vector2Int> result = new HashSet<Vector2Int>();
        int range = 1;

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i, j] != 0) {
                    for (int dx = -range; dx <= range; dx++) {
                        for (int dy = -range; dy <= range; dy++) 
                        {
                            int ni = i + dx, nj = j + dy;
                            if (InBounds(ni, nj) && board[ni, nj] == 0)
                                result.Add(new Vector2Int(ni, nj));
                        }
                    }
                }
            }
        }

        // Nếu bàn cờ rỗng, chọn ô giữa
        if (result.Count == 0)
            result.Add(new Vector2Int(4, 4));

        return new List<Vector2Int>(result);
    }
} 
