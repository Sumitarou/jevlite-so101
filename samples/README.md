# samples/

Only a small number of images that are safe to publish go here.

| Folder | Contents | Git |
|---|---|---|
| `synthetic/` | Synthetic scenes from `scripts/make_synthetic_samples.py` (no people). For pipeline and latency checks only, so their labels mean nothing. | committed |
| `real/` | Real photos around the arm that you have checked are safe to publish. **Never commit an image showing a face, a room, a monitor screen or any other personal information.** | committed |
| `private/` | Real photos that might contain personal information | ignored by `.gitignore` |

```bash
uv run python scripts/make_synthetic_samples.py
uv run python scripts/run_zeroshot_on_images.py --dir samples/
```
