
try:
    import flask
    with open('status.txt', 'w') as f:
        f.write('Flask OK')
except ImportError:
    with open('status.txt', 'w') as f:
        f.write('Flask Missing')
except Exception as e:
    with open('status.txt', 'w') as f:
        f.write(f'Error: {e}')
