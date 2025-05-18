
using UnityEngine;
using Unity.Barracuda;
using System;
using System.Collections.Generic;

public class MLBot : IDisposable
{
    private Model _runtimeModel;
    private IWorker _worker;

    public MLBot(NNModel modelAsset)
    {
        if (modelAsset == null)
        {
            Debug.LogError("Model asset is missing!");
            return;
        }
        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _runtimeModel);
    }

    // Dự đoán nước đi từ trạng thái bàn cờ
    public Vector2Int PredictMove(float[] boardState)
    {
        // Chuyển boardState thành tensor input
        Tensor inputTensor = new Tensor(1, 9, 9, 1, boardState);
        _worker.Execute(inputTensor);
        Tensor outputTensor = _worker.PeekOutput();

        // Tìm ô có xác suất cao nhất trong các ô TRỐNG
        int maxIndex = -1;
        float maxValue = float.MinValue;

        for (int i = 0; i < 81; i++)
        {
            int x = i / 9;
            int y = i % 9;

            // Chỉ xét ô trống (boardState[i] == 0)
            if (boardState[i] == 0 && outputTensor[i] > maxValue)
            {
                maxValue = outputTensor[i];
                maxIndex = i;
            }
        }

        // Nếu không có ô trống, chọn ngẫu nhiên (fallback)
        if (maxIndex == -1)
        {
            Debug.LogWarning("No valid moves. Using random.");
            return GetRandomEmptyCell(boardState);
        }

        inputTensor.Dispose();
        outputTensor.Dispose();

        return new Vector2Int(maxIndex / 9, maxIndex % 9);
    }

    // Hàm chọn ô trống ngẫu nhiên
    private Vector2Int GetRandomEmptyCell(float[] boardState)
    {
        List<Vector2Int> emptyCells = new List<Vector2Int>();
        for (int i = 0; i < 81; i++)
        {
            if (boardState[i] == 0)
            {
                emptyCells.Add(new Vector2Int(i / 9, i % 9));
            }
        }
        return emptyCells[UnityEngine.Random.Range(0, emptyCells.Count)];
    }

    public void Dispose()
    {
        _worker?.Dispose();
    }
}