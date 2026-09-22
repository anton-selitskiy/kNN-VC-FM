.PHONY: site-audio site-check site-serve

site-audio:
	@test -n "$(EVAL_ROOT)" || (echo "Set EVAL_ROOT to the evaluation-audio directory" && exit 1)
	python3 scripts/prepare_project_page_audio.py --eval-root "$(EVAL_ROOT)"

site-check:
	python3 -m py_compile scripts/prepare_project_page_audio.py
	@test -f dist/index.html
	@test -f dist/styles.css
	@test -f dist/app.js
	@test -f dist/audio/manifest.json

site-serve:
	python3 -m http.server 8000 --directory dist
