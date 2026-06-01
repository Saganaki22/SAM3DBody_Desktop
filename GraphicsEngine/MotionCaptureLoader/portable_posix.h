#ifndef FSB_PORTABLE_POSIX_H_INCLUDED
#define FSB_PORTABLE_POSIX_H_INCLUDED

#if defined(_WIN32)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef intptr_t ssize_t;

static inline ssize_t fsb_getline(char **lineptr, size_t *n, FILE *stream)
{
    if (!lineptr || !n || !stream) return -1;

    if (*lineptr == NULL || *n == 0)
    {
        *n = 256;
        *lineptr = (char*) malloc(*n);
        if (*lineptr == NULL) return -1;
    }

    size_t pos = 0;
    int ch = 0;
    while ((ch = fgetc(stream)) != EOF)
    {
        if (pos + 1 >= *n)
        {
            size_t next = (*n) * 2;
            char *grown = (char*) realloc(*lineptr, next);
            if (grown == NULL) return -1;
            *lineptr = grown;
            *n = next;
        }
        (*lineptr)[pos++] = (char) ch;
        if (ch == '\n') break;
    }

    if (pos == 0 && ch == EOF) return -1;
    (*lineptr)[pos] = '\0';
    return (ssize_t) pos;
}

#define getline fsb_getline
#endif

#endif
