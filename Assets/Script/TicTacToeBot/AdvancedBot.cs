using System.Collections.Generic;
using UnityEngine;
using System;

public class AdvancedBot : IBotStrategy
{
    private int maxTime; // Maximum milliseconds to search
    private int maxDepth; // Maximum depth to search in any iteration
    private const int winScore = 1000000;
    private const int fourInRowScore = 100000;
    private const int threeInRowScore = 1000;
    private const int twoInRowScore = 100;
    private const int winLength = 5;

    // Additional position evaluation factors
    private const int centerControlBonus = 10; 
    private const int mobilityBonus = 5;
    private const int threatBonus = 20;
    
    private System.Diagnostics.Stopwatch stopwatch;
    private bool timeUp;
    private int nodesEvaluated;
    private Vector2Int bestMove;

    public AdvancedBot(int maxSearchTimeMs = 500, int maxDepth = 15)
    {
        this.maxTime = maxSearchTimeMs;
        this.maxDepth = maxDepth;
        this.stopwatch = new System.Diagnostics.Stopwatch();
    }

    public Vector2Int GetNextMove(int[,] board, int currentTurn)
    {
        // If this is the first move and center is available, take it
        if (IsEmptyBoard(board))
        {
            return new Vector2Int(4, 4);
        }

        timeUp = false;
        stopwatch.Restart();
        bestMove = Vector2Int.zero;
        nodesEvaluated = 0;
        
        var immediateWin = FindWinningMove(board, currentTurn);
        if (immediateWin != null)
        {
            return immediateWin.Value;
        }
        
        var blockMove = FindWinningMove(board, -currentTurn);
        if (blockMove != null)
        {
            return blockMove.Value;
        }

        // iterative deepening
        List<Vector2Int> moves = GetOrderedCandidateMoves(board, currentTurn);
        if (moves.Count == 0)
            return new Vector2Int(4, 4); // Fallback to center if no moves found
            
        // Start with the best move from previous searches - typically the center early on
        bestMove = moves[0];
        
        // Perform iterative deepening search
        for (int depth = 2; depth <= maxDepth; depth += 1)
        {
            if (timeUp) break;
            
            int bestScore = int.MinValue;
            int alpha = int.MinValue;
            int beta = int.MaxValue;
            Vector2Int currentBestMove = bestMove; // Initialize with previous best
            bool moveFound = false;
            
            // Randomize the first few moves a bit to add variety
            if (CountMoves(board) < 10 && UnityEngine.Random.Range(0, 100) < 20)
            {
                // Sometimes apply a small random shuffle to the first few moves
                for (int i = 0; i < Math.Min(3, moves.Count - 1); i++)
                {
                    int j = UnityEngine.Random.Range(i, Math.Min(i + 2, moves.Count));
                    if (i != j)
                    {
                        Vector2Int temp = moves[i];
                        moves[i] = moves[j];
                        moves[j] = temp;
                    }
                }
            }
            
            foreach (Vector2Int move in moves)
            {
                if (timeUp) break;
                
                // Try the move
                board[move.x, move.y] = currentTurn;
                int score = -AlphaBeta(board, depth - 1, -beta, -alpha, -currentTurn);
                board[move.x, move.y] = 0; // Undo move
                
                if (score > bestScore)
                {
                    bestScore = score;
                    currentBestMove = move;
                    moveFound = true;
                    
                    // Update alpha
                    alpha = Math.Max(alpha, score);
                }
                
                // Check if time is up
                if (stopwatch.ElapsedMilliseconds > maxTime)
                {
                    timeUp = true;
                    break;
                }
            }
            
            if (moveFound)
                bestMove = currentBestMove;
                
            // If we found a winning score no need to search deeper
            if (bestScore >= winScore / 2)
                break;
                
            // If we've run out of time use the best move found so far
            if (timeUp)
                break;
        }
        
        //Debug.Log($"AdvancedBot evaluated {nodesEvaluated} nodes in {stopwatch.ElapsedMilliseconds}ms");
        stopwatch.Stop();
        return bestMove;
    }
    
    private int AlphaBeta(int[,] board, int depth, int alpha, int beta, int player)
    {
        nodesEvaluated++;
        
        // Check if time is up
        if (stopwatch.ElapsedMilliseconds > maxTime)
        {
            timeUp = true;
            return EvaluateBoard(board, player);
        }
        
        // Check for terminal conditions or max depth
        if (depth <= 0)
            return EvaluateBoard(board, player);
            
        // Check for win/loss
        int gameResult = CheckWin(board);
        if (gameResult != 0)
            return gameResult == player ? winScore : -winScore;
            
        // Check for draw
        if (IsBoardFull(board))
            return 0;
        
        List<Vector2Int> moves = GetOrderedCandidateMoves(board, player);
        
        // Use quiescence search for leaf nodes
        if (depth <= 2 && !IsBoardQuiet(board, player))
        {
            // If board is not quiet at the leaf node, extend search
            depth += 1;
        }
        
        int bestScore = int.MinValue;
        
        foreach (Vector2Int move in moves)
        {
            if (timeUp) break;
            
            board[move.x, move.y] = player;
            int score = -AlphaBeta(board, depth - 1, -beta, -alpha, -player);
            board[move.x, move.y] = 0; // Undo move
            
            bestScore = Math.Max(bestScore, score);
            alpha = Math.Max(alpha, bestScore);
            
            // Alpha-beta cutoff
            if (alpha >= beta)
                break;
        }
        
        return bestScore;
    }
    
    private int EvaluateBoard(int[,] board, int player)
    {
        int score = 0;
        
        // Material advantage - count pieces
        int myPieces = 0, opponentPieces = 0;
        
        // Pattern recognition - evaluate all lines
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] == player) 
                    myPieces++;
                else if (board[i, j] == -player)
                    opponentPieces++;
                
                if (board[i, j] == 0) continue;
                
                int currentPlayer = board[i, j];
                int multiplier = (currentPlayer == player) ? 1 : -1;
                
                // Evaluate lines in all 4 directions
                score += EvaluateLine(board, i, j, 1, 0, currentPlayer) * multiplier; // Horizontal
                score += EvaluateLine(board, i, j, 0, 1, currentPlayer) * multiplier; // Vertical
                score += EvaluateLine(board, i, j, 1, 1, currentPlayer) * multiplier; // Diagonal down-right
                score += EvaluateLine(board, i, j, 1, -1, currentPlayer) * multiplier; // Diagonal up-right
            }
        }
        
        // Center control
        if (board[4, 4] == player) score += centerControlBonus;
        else if (board[4, 4] == -player) score -= centerControlBonus;
        
        // Mobility - value having more moves available
        int myMobility = GetMobilityScore(board, player);
        int opponentMobility = GetMobilityScore(board, -player);
        score += (myMobility - opponentMobility) * mobilityBonus / 10;

        // Randomization for equal positions to enhance unpredictability
        score += UnityEngine.Random.Range(-10, 11);
        return score;
    }
    
    private int GetMobilityScore(int[,] board, int player)
    {
        int score = 0;
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0) continue;
                
                // Check if this empty spot is valuable for the player
                bool hasAdjacentPiece = false;
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = i + dx;
                        int ny = j + dy;
                        
                        if (nx >= 0 && nx < 9 && ny >= 0 && ny < 9 && board[nx, ny] == player)
                        {
                            hasAdjacentPiece = true;
                            break;
                        }
                    }
                    
                    if (hasAdjacentPiece) break;
                }
                
                if (hasAdjacentPiece) score++;
            }
        }
        
        return score;
    }
    
    private int EvaluateLine(int[,] board, int x, int y, int dx, int dy, int player)
    {
        int score = 0;
        
        // Check all possible 5-in-a-row sequences that include this cell
        for (int k = 0; k < winLength; k++)
        {
            int count = 0;
            int empty = 0;
            int blocked = 0;
            
            for (int l = 0; l < winLength; l++)
            {
                int nx = x + (l - k) * dx;
                int ny = y + (l - k) * dy;
                
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9)
                {
                    blocked++;
                    break;
                }
                
                if (board[nx, ny] == player) count++;
                else if (board[nx, ny] == 0) empty++;
                else
                {
                    blocked++;
                    break;
                }
            }
            
            // Evaluate the pattern
            if (blocked == 0)
            {
                // Winning line
                if (count == winLength) return winScore;
                
                // Potential lines with various scores based on pattern
                if (count == 4 && empty == 1) score += fourInRowScore;
                else if (count == 3 && empty == 2) score += threeInRowScore;
                else if (count == 2 && empty == 3) score += twoInRowScore;
                else if (count == 1 && empty == 4) score += 10;
            }
            else if (blocked == 1 && empty + count == winLength)
            {
                // Semi-open sequences also have value but less
                if (count == 4) score += fourInRowScore / 4;
                else if (count == 3) score += threeInRowScore / 4;
                else if (count == 2) score += twoInRowScore / 4;
            }
        }
        
        return score;
    }
    
    private List<Vector2Int> GetOrderedCandidateMoves(int[,] board, int player)
    {
        // Using a hashset to avoid duplicates
        var candidateMoves = new List<Vector2Int>();
        var moveScores = new Dictionary<Vector2Int, int>();
        
        // Check for critical squares first - win opportunities or blocking opponent wins
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0) continue;
                
                // Try the move for the current player
                board[i, j] = player;
                if (CheckWin(board) == player)
                {
                    board[i, j] = 0; // Undo move
                    // This is a winning move, prioritize it
                    return new List<Vector2Int> { new Vector2Int(i, j) };
                }
                board[i, j] = 0; // Undo move
                
                // Try the move for the opponent
                board[i, j] = -player;
                if (CheckWin(board) == -player)
                {
                    board[i, j] = 0; // Undo move
                    // This is a blocking move, prioritize it
                    candidateMoves.Insert(0, new Vector2Int(i, j));
                    moveScores[new Vector2Int(i, j)] = int.MaxValue;
                    continue;
                }
                board[i, j] = 0; // Undo move
                
                // Check if this is a good candidate square
                bool isCandidate = false;
                int moveScore = 0;
                
                // Check neighborhood - only consider squares near existing pieces
                for (int dx = -2; dx <= 2; dx++)
                {
                    for (int dy = -2; dy <= 2; dy++)
                    {
                        int nx = i + dx;
                        int ny = j + dy;
                        
                        if (nx >= 0 && nx < 9 && ny >= 0 && ny < 9 && board[nx, ny] != 0)
                        {
                            isCandidate = true;
                            // Squares closer to existing pieces are better
                            moveScore += (3 - Math.Max(Math.Abs(dx), Math.Abs(dy))) * 10;
                            
                            // Squares near our pieces are better than near opponent's pieces
                            if (board[nx, ny] == player)
                                moveScore += 5;
                        }
                    }
                }
                
                // Quick pattern evaluation for determining move priority
                if (isCandidate)
                {
                    // Try move for us
                    board[i, j] = player;
                    int ourScore = QuickEvaluatePosition(board, i, j, player);
                    board[i, j] = 0;
                    
                    // Try move for opponent
                    board[i, j] = -player;
                    int theirScore = QuickEvaluatePosition(board, i, j, -player);
                    board[i, j] = 0;
                    
                    // Combine scores
                    moveScore += ourScore + theirScore/2; // Prioritize our threats slightly
                    
                    // Add move with score
                    var move = new Vector2Int(i, j);
                    candidateMoves.Add(move);
                    moveScores[move] = moveScore;
                }
            }
        }
        
        // If no candidates, check if board is empty or consider all empty squares
        if (candidateMoves.Count == 0)
        {
            if (IsEmptyBoard(board))
            {
                return new List<Vector2Int> { new Vector2Int(4, 4) }; // Start at center
            }
            
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    if (board[i, j] == 0)
                    {
                        var move = new Vector2Int(i, j);
                        candidateMoves.Add(move);
                        moveScores[move] = 0;
                    }
                }
            }
        }
        
        // Sort moves by score
        candidateMoves.Sort((a, b) => moveScores[b].CompareTo(moveScores[a]));
        
        // Limit to top N moves to improve performance
        int maxMoves = Math.Min(12, candidateMoves.Count);
        return candidateMoves.GetRange(0, maxMoves);
    }
    
    private int QuickEvaluatePosition(int[,] board, int x, int y, int player)
    {
        int score = 0;
        
        // Evaluate patterns in all 4 directions
        score += EvaluateThreat(board, x, y, 1, 0, player);  // Horizontal
        score += EvaluateThreat(board, x, y, 0, 1, player);  // Vertical
        score += EvaluateThreat(board, x, y, 1, 1, player);  // Diagonal down-right
        score += EvaluateThreat(board, x, y, 1, -1, player); // Diagonal up-right
        
        return score;
    }
    
    private int EvaluateThreat(int[,] board, int x, int y, int dx, int dy, int player)
    {
        int score = 0;
        
        // Check for various patterns (open threes, fours, etc)
        for (int offset = -4; offset <= 0; offset++)
        {
            int count = 0;
            int empty = 0;
            bool valid = true;
            
            for (int k = 0; k < winLength; k++)
            {
                int nx = x + (offset + k) * dx;
                int ny = y + (offset + k) * dy;
                
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 9)
                {
                    valid = false;
                    break;
                }
                
                if (board[nx, ny] == player) count++;
                else if (board[nx, ny] == 0) empty++;
                else
                {
                    valid = false;
                    break;
                }
            }
            
            if (valid)
            {
                // Open four (one move from win)
                if (count == 4 && empty == 1)
                    score += 1000;
                // Open three (two moves from win, but threatening)
                else if (count == 3 && empty == 2)
                    score += 100;
                // Open two (potential threat)
                else if (count == 2 && empty == 3)
                    score += 10;
            }
        }
        
        return score;
    }
    
    private int CheckWin(int[,] board)
    {
        // Check for a win in any direction
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] == 0) continue;
                
                int player = board[i, j];
                
                // Check horizontal
                if (i <= 9 - winLength)
                {
                    bool win = true;
                    for (int k = 1; k < winLength; k++)
                    {
                        if (board[i + k, j] != player)
                        {
                            win = false;
                            break;
                        }
                    }
                    if (win) return player;
                }
                
                // Check vertical
                if (j <= 9 - winLength)
                {
                    bool win = true;
                    for (int k = 1; k < winLength; k++)
                    {
                        if (board[i, j + k] != player)
                        {
                            win = false;
                            break;
                        }
                    }
                    if (win) return player;
                }
                
                // Check diagonal down-right
                if (i <= 9 - winLength && j <= 9 - winLength)
                {
                    bool win = true;
                    for (int k = 1; k < winLength; k++)
                    {
                        if (board[i + k, j + k] != player)
                        {
                            win = false;
                            break;
                        }
                    }
                    if (win) return player;
                }
                
                // Check diagonal up-right
                if (i <= 9 - winLength && j >= winLength - 1)
                {
                    bool win = true;
                    for (int k = 1; k < winLength; k++)
                    {
                        if (board[i + k, j - k] != player)
                        {
                            win = false;
                            break;
                        }
                    }
                    if (win) return player;
                }
            }
        }
        
        return 0; // No winner
    }
    
    private bool IsBoardFull(int[,] board)
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] == 0)
                    return false;
            }
        }
        return true;
    }
    
    private bool IsEmptyBoard(int[,] board)
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0)
                    return false;
            }
        }
        return true;
    }
    
    private int CountMoves(int[,] board)
    {
        int count = 0;
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0)
                    count++;
            }
        }
        return count;
    }
    
    private bool IsBoardQuiet(int[,] board, int player)
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0) continue;
                
                board[i, j] = player;
                int ourScore = QuickEvaluatePosition(board, i, j, player);
                board[i, j] = -player;
                int theirScore = QuickEvaluatePosition(board, i, j, -player);
                board[i, j] = 0;
                
                if (ourScore >= 1000 || theirScore >= 1000)
                    return false; 
            }
        }
        
        return true; 
    
    private Vector2Int? FindWinningMove(int[,] board, int player)
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i, j] != 0) continue;
                
                board[i, j] = player;
                bool isWin = CheckWin(board) == player;
                board[i, j] = 0; // Undo move
                
                if (isWin)
                    return new Vector2Int(i, j);
            }
        }
        
        return null; 
    }
}