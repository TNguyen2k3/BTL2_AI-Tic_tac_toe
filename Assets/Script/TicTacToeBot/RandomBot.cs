using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomBot : IBotStrategy
{
    public Vector2Int GetNextMove(int[,] board, int currentTurn)
    {
        List<Vector2Int> emptyCells = new List<Vector2Int>();
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (board[i, j] == 0)
                    emptyCells.Add(new Vector2Int(i, j));

        return emptyCells[Random.Range(0, emptyCells.Count)];
    }
}

