all:
	make -C layers/cpp/
test:
	nosetests -v tests/*.py
