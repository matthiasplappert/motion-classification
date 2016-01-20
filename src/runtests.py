from unittest import (TestLoader, TextTestRunner)

if __name__ == '__main__':
    suite = TestLoader().discover('tests')
    runner = TextTestRunner()
    runner.run(suite)
