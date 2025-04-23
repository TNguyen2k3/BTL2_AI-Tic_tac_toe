using UnityEngine;

public interface IBotStrategy
{
    Vector2Int GetNextMove(int[,] board, int currentTurn);
}
