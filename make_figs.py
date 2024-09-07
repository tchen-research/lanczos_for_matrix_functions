import subprocess

files = [
    'CG_MINRES',
    'CG_spec_adapt',
    'cheb_vs_monomial',
    'fp_greenbaum',
    'fp_QfTQv',
    'GQ_dist_interlace',
    'greenbaum_convergence',
    'Lanczos-FA_opt',
    'legend'
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)