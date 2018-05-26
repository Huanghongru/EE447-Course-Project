import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

def parse_lablog(logfile):
    """
    Parse experiment log file to obtain results

    Parameters:
        logfile: (string) name of the log file.
    """
    qs = []
    hd = []
    ci = []
    kcore = []
    pagerank = []
    fs = []
    with open(logfile, 'r') as f:
        for line in f.readlines():
            if line[0] == 'q':
                q, result = line.split('\t')
                qs.append(eval(q[3:]))
                result = eval(result[8:])

                hd.append(result['hd'] if result.get('hd') else 0)
                ci.append(result['ci'] if result.get('ci') else 0)
                kcore.append(result['kcore'] if result.get('kcore') else 0)
                pagerank.append(result['pagerank'] if result.get('pagerank') else 0)
                fs.append(result['fanshen'] if result.get('fanshen') else 0)

    return (np.array(qs), np.array(hd), np.array(ci), 
        np.array(kcore), np.array(pagerank), np.array(fs))

def draw_q_vs_spreadRange(qs, hd, ci, kc, pr, fs):
    """
    Experiment result visualization. 
    """
    color = {'red':'#f34236', 'purple': '#550f9d', 
    'yellow':'#ffc800', 'green':'#228b22', 'blue':'#0081cc'}

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel('q', weight='bold')
    ax.set_ylabel('spread range', weight='bold')

    hd_plot = ax.plot(qs, hd, marker='.', markersize=3, color=color['red'])
    pr_plot = ax.plot(qs, pr, marker='.', markersize=3, color=color['yellow'])
    kc_plot = ax.plot(qs, kc, marker='.', markersize=3, color=color['green'])
    ci_plot = ax.plot(qs, ci, marker='.', markersize=3, color=color['blue'])
    fs_plot = ax.plot(qs, fs, marker='.', markersize=3, color=color['purple'])

    plt.legend(("HD", "PageRank", "K-core", "CI", "fanshen"), loc=4)
    plt.show()

def main():
    qs, hd, ci, kc, pr, fs = parse_lablog('pow.log')
    draw_q_vs_spreadRange(qs, hd, ci, kc, pr, fs)

if __name__ == '__main__':
    main()
