#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config_file.h"

// config file full-name with path
static char g_filename[FILENAME_MAX_LEN] = {""};

// remove ' and " in header and tail
// remove \t\r\n in tail
void string_remove_mark(char* arg)
{
    if(arg != NULL)
    {
        int len = (int) strlen(arg);
        while(len>0 && ('\n' == arg[len-1] || '\r' == arg[len-1] || '\t' == arg[len-1]))
        {
            arg[len-1] = '\0';
            len --;
        }
        if(len >= 2)
        {
            if('\"' == arg[len-1] || '\'' == arg[len-1])
            {
                arg[len-1] = '\0';
                len--;
            }
            if('\"' == arg[0] || '\'' == arg[0])
            {
                memmove(arg, arg+1, len+1);
            }
        }
    }
}

// setting config file-name,  must by called before other functions
int config_file_init(const char* filename)
{
    if(filename != NULL && filename[0] != '\0' && strlen(filename) < FILENAME_MAX_LEN)
    {
        strcpy(g_filename, filename);
        return 1;
    }
    memset(g_filename, '\0', sizeof(g_filename));
    return 0;
}

int config_file_get(const char* key, char* value, int value_size, const char* filename)
{
	FILE* fp = NULL;
    int result = 0;
    if(value == NULL || value_size < 1)
    {
        return 0;
    }
    *value = 0;

    if(key == NULL || key[0] == '\0' || value_size < 2)
    {
        return 0;
    }
    if(strlen(key) >= KEY_VALUE_MAX_LEN)
    {
        return 0;
    }
	if(NULL != (fp = fopen(filename==NULL?g_filename:filename, "r")))
	{
        char buf[(KEY_VALUE_MAX_LEN+2)*2] = {""};
        char dst[KEY_VALUE_MAX_LEN+2] = {""};
		strcpy(dst, key);
		strcat(dst, "=");
		while(NULL != fgets(buf, sizeof(buf)-1, fp))
		{
            buf[sizeof(buf)-1] = '\0';
			if(buf[0] != '#' && buf[0] != ';' && // ignore note-line
               buf == strstr(buf, dst))
			{
                int value_len = 0;
                char* value_ptr = NULL;
				string_remove_mark(buf);
                value_ptr = buf + strlen(dst);
                value_len = (int) strlen(value_ptr);
                if(value_len >= value_size)
                {
                    return 0; // value buffer overflow
                }
				strcpy(value, value_ptr);
                result = value_len;
				break;
			}
		}
		fclose(fp);
	}
    return result;
}

int config_file_set(const char* key, const char* value, const char* filename)
{
	FILE* fp = NULL;
    char* tmp = NULL;
    int is_found_key = 0;
    if(key == NULL || key[0] == '\0' || value == NULL)
    {
        return 0;
    }
    if(strlen(key) >= KEY_VALUE_MAX_LEN || strlen(value) >= KEY_VALUE_MAX_LEN)
    {
        return 0;
    }
	if(NULL != (fp = fopen(filename==NULL?g_filename:filename, "r")))
	{
        char buf[(KEY_VALUE_MAX_LEN+2)*2] = {""};
        char dst[KEY_VALUE_MAX_LEN+2] = {""};
        size_t maxlen = 0;
		size_t len = 0;
		fseek(fp, 0, SEEK_END);
		maxlen = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		tmp = (char*) malloc(maxlen + (KEY_VALUE_MAX_LEN+2)*2);
		if(NULL == tmp)
		{
			return 0;
		}
		strcpy(tmp, "");

		strcpy(dst, key);
		strcat(dst, "=");
		while(NULL != fgets(buf, sizeof(buf), fp))
		{
			len = strlen(buf);
			if(len > 0)
			{
                if(buf == strstr(buf, dst))
                {
                    // found this key, replace with new value
                    is_found_key = 1;
                    sprintf(tmp + strlen(tmp), "%s=%s\n", key, value);
                }
                else
                {
                    strcat(tmp, buf);
                    if('\n' != buf[len-1] && '\r' != buf[len-1])
                    {
                        strcat(tmp, "\n");
                    }
                }
			}
		}
		fclose(fp);

        // not found this key, add new line
        if(!is_found_key)
        {
            sprintf(tmp + strlen(tmp), "%s=%s\n", key, value);
        }
	}
    if(tmp == NULL)
    {
        return 0;
    }
	if(NULL == (fp = fopen(filename==NULL?g_filename:filename, "w")))
    {
        // failed to open file with write-mode
        free(tmp);
        return 0;
    }
    else
    {
        fputs(tmp, fp); // write to file
		fclose(fp);
	}
    free(tmp);
    return 1;
}

int config_file_getall(char* buffer, int buffer_size, const char* filename)
{
	FILE* fp = NULL;
	int result = 0;
	if(buffer != NULL && buffer_size > 5)
	{
		if(NULL != (fp = fopen(filename==NULL?g_filename:filename, "rb")))
		{
			const int read_len = fread(buffer, 1, buffer_size-1, fp);
			if(read_len > 0)
			{
				buffer[read_len] = '\0';
				result = read_len;
			}
			fclose(fp);
			fp = NULL;
		}
	}
	return result;
}

int config_file_setall(const char* buffer, const char* filename)
{
	FILE* fp = NULL;
	int result = 0;
	if(buffer != NULL)
	{
		const size_t buffer_size = strlen(buffer);
		if(buffer_size > 0)
		{
			if(NULL != (fp = fopen(filename==NULL?g_filename:filename, "wb")))
			{
				result = fwrite(buffer, 1, buffer_size, fp);
				fclose(fp);
				fp = NULL;
			}
		}
	}
	return result;
}

int config_file_get_int(const char* key, int default_value, const char* filename)
{
    char value[KEY_VALUE_MAX_LEN] = {""};
    if(config_file_get(key, value, (int)sizeof(value), filename) <= 0)
    {
        return default_value;
    }
    if(value[0] == '0' && (value[1] == 'x' || value[1] == 'X'))
    {
        //hex
        unsigned int result = 0;
        if(1 != sscanf(value + 2, "%X", &result))
        {
            return default_value;
        }
        return (int)result;
    }
    return atoi(value);
}

const char* config_file_get_str(const char* key, char* value, int value_size, const char* default_value, const char* filename)
{
    if(config_file_get(key, value, value_size, filename) <= 0)
    {
        return default_value;
    }
    return value;
}
