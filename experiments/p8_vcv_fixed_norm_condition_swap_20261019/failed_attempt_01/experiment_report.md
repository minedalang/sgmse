# Failed launch, attempt 01

Failed before model load; frontend/backend/recognizer forwards all zero. NumPy float32 bounds were not JSON serializable, including during failure reporting. Original driver, helper, config, outputs and log are preserved here. The separately recorded serialization-only repair casts both stored bounds to Python float. No scientific settings changed.
