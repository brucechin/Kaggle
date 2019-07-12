#encoding=utf-8
#你好
match_award      = 1
mismatch_penalty = -1
gap_penalty      = -2

def zero_matrix(xsize,ysize):#return zero matrix whose shape is (xsize,ysize)
    return [[0]*ysize for i in range(xsize)]

def match_score(left,right):
    if(left == right):
        return match_award
    elif left == '-' or right == '-':
        return gap_penalty
    else:
        return mismatch_penalty


def finalize(align1, align2):
    align1 = align1[::-1]  # reverse sequence 1
    align2 = align2[::-1]  # reverse sequence 2

    i, j = 0, 0

    # calcuate score and aligned sequeces
    symbol = ''
    found = 0
    score = 0
    for i in range(0, len(align1)):
        # if two AAs are the same, then output the letter
        if align1[i] == align2[i]:
            symbol = symbol + align1[i]
            score += match_score(align1[i], align2[i])

        # if they are not identical and none of them is gap
        elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-':
            score += match_score(align1[i], align2[i])
            symbol += ' '
            found = 0

        # if one of them is a gap, output a space
        elif align1[i] == '-' or align2[i] == '-':
            symbol += ' '
            score += gap_penalty

    print('Score =', score)
    print(align1)
    print(symbol)
    print(align2)


def global_alignment(seq1,seq2):
    m,n = len(seq1),len(seq2)
    score = zero_matrix(m+1,n+1)

    #initialization for gap penalty
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j

    #calculate the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + match_score(seq1[i - 1], seq2[j - 1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)

    #trace back from the bottom right to top left, then get aligned seqences
    align1, align2 = '', ''
    i, j = m, n  # start from the bottom right cell

    while i > 0 and j > 0:  # end toching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i - 1][j - 1]
        score_up = score[i][j - 1]
        score_left = score[i - 1][j]

        if score_current == score_diagonal + match_score(seq1[i - 1], seq2[j - 1]):#go left up
            align1 += seq1[i - 1]
            align2 += seq2[j - 1]
            i -= 1
            j -= 1
        elif score_current == score_left + gap_penalty:#go left
            align1 += seq1[i - 1]
            align2 += '-'
            i -= 1
        elif score_current == score_up + gap_penalty:#go up
            align1 += '-'
            align2 += seq2[j - 1]
            j -= 1

        # Finish tracing up to the top left cell
    while i > 0:
        align1 += seq1[i - 1]
        align2 += '-'
        i -= 1
    while j > 0:
        align1 += '-'
        align2 += seq2[j - 1]
        j -= 1

    finalize(align1, align2)

a = 'AGCTGATCGATTGCTAGCTAGCTATAGATCTAG'
b = 'TAGCTTAGCTAGATCGAGATCGATCTGATCGATCATGCTAG'

global_alignment(a,b)