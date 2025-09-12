import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Box, Typography, Tabs, Tab, Card, CardContent, Chip, IconButton, CircularProgress, Alert, useTheme, alpha, Stack, Tooltip, Badge, useMediaQuery, Fab, Divider, Pagination, TextField, InputAdornment, Button, Dialog, DialogTitle, DialogContent, DialogActions, DialogContentText, Checkbox, Collapse, Switch, Avatar } from '@mui/material';
import { CheckCircle, Warning as WarningIcon, Delete as DeleteIcon, Refresh as RefreshIcon, Assignment as AssignmentIcon, ExpandMore as ExpandMoreIcon, ExpandLess as ExpandLessIcon, Schedule as ScheduleIcon, SkipNext as SkipIcon, Storage as DatabaseIcon, Timeline as OperationsIcon, Source as SourceIcon, Folder as FolderIcon, Movie as MovieIcon, Tv as TvIcon, InsertDriveFile as FileIcon, PlayCircle as PlayCircleIcon, FolderOpen as FolderOpenIcon, Info as InfoIcon, CheckCircle as ProcessedIcon, RadioButtonUnchecked as UnprocessedIcon, Link as LinkIcon, Warning as WarningIcon2, Settings as SettingsIcon, Search as SearchIcon, DeleteSweep as DeleteSweepIcon, FlashAuto as AutoModeIcon, PlayArrow as PlayArrowIcon, Restore as RestoreIcon } from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import DatabaseSearch from './DatabaseSearch';
import ProcessingAnimation from './ProcessingAnimation';
import { useSSEEventListener } from '../../hooks/useCentralizedSSE';
import { FileItem } from '../FileBrowser/types';
import ModifyDialog from '../FileBrowser/ModifyDialog/ModifyDialog';
import ManualImport from './ManualImport';


const MotionFab = motion(Fab);

interface FileOperation {
  id: string;
  filePath: string;
  destinationPath?: string;
  fileName: string;
  status: 'created' | 'failed' | 'deleted' | 'skipped';
  timestamp: string;
  reason?: string;
  error?: string;
  tmdbId?: string;
  seasonNumber?: number;
  type: 'movie' | 'tvshow' | 'other';
  operation?: 'process' | 'delete';
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`file-operations-tabpanel-${index}`}
      aria-labelledby={`file-operations-tab-${index}`}
      style={{ display: value === index ? 'block' : 'none' }}
      {...other}
    >
      <Box sx={{ pt: 3 }}>{children}</Box>
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `file-operations-tab-${index}`,
    'aria-controls': `file-operations-tabpanel-${index}`,
  };
}

function FileOperations() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [mainTabValue, setMainTabValue] = useState(0);
  const [tabValue, setTabValue] = useState(0);
  const [operations, setOperations] = useState<FileOperation[]>([]);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [successDialogOpen, setSuccessDialogOpen] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Auto-close success dialog after 3 seconds
  useEffect(() => {
    if (successDialogOpen) {
      const timer = setTimeout(() => {
        setSuccessDialogOpen(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [successDialogOpen]);

  // URL-synchronized pagination for operations
  const currentPageFromUrl = parseInt(searchParams.get('page') || '1', 10);
  const [currentPageState, setCurrentPageState] = useState(currentPageFromUrl);
  const [recordsPerPage] = useState(25);
  const [totalOperations, setTotalOperations] = useState(0);
  const [statusCounts, setStatusCounts] = useState({
    created: 0,
    failed: 0,
    skipped: 0,
    deleted: 0,
  });

  // URL-synchronized pagination for source files
  const sourcePageFromUrl = parseInt(searchParams.get('sourcePage') || '1', 10);
  const [sourcePageState, setSourcePageState] = useState(sourcePageFromUrl);

  // Functions to update both state and URL
  const setCurrentPage = useCallback((newPage: number) => {
    setCurrentPageState(newPage);
    const newSearchParams = new URLSearchParams(searchParams);
    if (newPage === 1) {
      newSearchParams.delete('page');
    } else {
      newSearchParams.set('page', newPage.toString());
    }
    setSearchParams(newSearchParams, { replace: true });
  }, [searchParams, setSearchParams]);

  const setSourcePage = useCallback((newPage: number) => {
    setSourcePageState(newPage);
    const newSearchParams = new URLSearchParams(searchParams);
    if (newPage === 1) {
      newSearchParams.delete('sourcePage');
    } else {
      newSearchParams.set('sourcePage', newPage.toString());
    }
    setSearchParams(newSearchParams, { replace: true });
  }, [searchParams, setSearchParams]);

  const currentPage = currentPageState;
  const sourcePage = sourcePageState;

  // Source File Browser state
  const [sourceFiles, setSourceFiles] = useState<FileItem[]>([]);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [sourceError, setSourceError] = useState('');
  const [sourceIndex] = useState<number | undefined>(undefined);
  const [sourceTotalPages, setSourceTotalPages] = useState(1);
  const [sourceTotalFiles, setSourceTotalFiles] = useState(0);
  const [modifyDialogOpen, setModifyDialogOpen] = useState(false);
  const [currentFileForProcessing, setCurrentFileForProcessing] = useState<string>('');
  const [hasSourceDirectories, setHasSourceDirectories] = useState<boolean | null>(null);

  // Processing animation state
  const [processingFiles, setProcessingFiles] = useState<Map<string, {
    fileName: string;
    mediaName?: string;
    mediaType?: string;
  }>>(new Map());

  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [sourceSearchQuery, setSourceSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [isSourceSearching, setIsSourceSearching] = useState(false);

  const searchQueryRef = useRef('');
  const sourceSearchQueryRef = useRef('');

  // Bulk delete state
  const [bulkDeleteDialogOpen, setBulkDeleteDialogOpen] = useState(false);
  const [bulkDeleteLoading, setBulkDeleteLoading] = useState(false);

  // Bulk selection state for operations (Created, Failed, Skipped tabs)
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [bulkActionLoading, setBulkActionLoading] = useState(false);
  const [bulkModifyDialogOpen, setBulkModifyDialogOpen] = useState(false);
  const [bulkModifyFilePaths, setBulkModifyFilePaths] = useState<string[]>([]);
  const [bulkProcessingInProgress, setBulkProcessingInProgress] = useState(false);


  // Bulk selection state for source files
  const [selectedSourceFiles, setSelectedSourceFiles] = useState<Set<string>>(new Set());

  const [autoModeEnabled, setAutoModeEnabled] = useState(false);
  const [autoProcessingFiles, setAutoProcessingFiles] = useState<Set<string>>(new Set());
  const [bulkProcessing, setBulkProcessing] = useState(false);
  const [manualImportOpen, setManualImportOpen] = useState(false);

  const fetchAutoModeSetting = async () => {
    try {
      const response = await axios.get('/api/config');
      const config = response.data.config;
      const autoModeSetting = config.find((item: any) => item.key === 'FILE_OPERATIONS_AUTO_MODE');
      setAutoModeEnabled(autoModeSetting?.value === 'true');
    } catch (error) {
      setAutoModeEnabled(false);
    }
  };

  const updateAutoModeSetting = async (enabled: boolean) => {
    try {
      await axios.post('/api/config/update-silent', {
        updates: [{
          key: 'FILE_OPERATIONS_AUTO_MODE',
          value: enabled.toString(),
          type: 'boolean',
          required: false
        }]
      });
    } catch (error) {
    }
  };

  useEffect(() => {
    fetchAutoModeSetting();
  }, []);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('lg'));

  const ITEMS_PER_PAGE = 50;

  const filteredSourceFiles = sourceFiles;

  const fetchSourceFilesData = useCallback(async (pageNum?: number, sourceIndexFilter?: number) => {
    setSourceLoading(true);
    setSourceError('');
    try {
      const actualPageNum = pageNum || sourcePage;
      const actualSourceIndex = sourceIndexFilter !== undefined ? sourceIndexFilter : sourceIndex;

      const params = new URLSearchParams({
        limit: ITEMS_PER_PAGE.toString(),
        offset: ((actualPageNum - 1) * ITEMS_PER_PAGE).toString(),
        activeOnly: 'true',
        mediaOnly: 'false'
      });

      if (actualSourceIndex !== undefined) {
        params.append('sourceIndex', actualSourceIndex.toString());
      }

      const currentSearch = sourceSearchQueryRef.current;
      if (currentSearch && currentSearch.trim()) {
        params.append('search', currentSearch.trim());
      }

      const response = await axios.get(`/api/database/source-files?${params.toString()}`);

      if (response.status === 200) {
        const data = response.data;

        // Add comprehensive null/undefined checks
        if (!data) {
          setSourceError('Invalid response from server');
          setSourceFiles([]);
          return;
        }

        // Handle both null and empty array cases for files
        let filesArray = data.files;
        if (filesArray === null || filesArray === undefined) {
          filesArray = [];
        } else if (!Array.isArray(filesArray)) {
          setSourceError('Invalid file data from server');
          setSourceFiles([]);
          return;
        }

        // Empty files array is valid - it means all files are processed or no files found
        if (filesArray.length === 0) {
          setSourceFiles([]);
          setSourceTotalPages(data.totalPages || 1);
          setSourceTotalFiles(data.total || 0);
          setHasSourceDirectories(true);
          setError('');
          return;
        }

        // Convert database format to FileItem format for compatibility
        const convertedFiles: FileItem[] = filesArray.map((dbFile: any) => ({
          name: dbFile.fileName || 'Unknown',
          path: dbFile.relativePath || '',
          fullPath: dbFile.filePath || '',
          type: 'file',
          size: dbFile.fileSizeFormatted || '0 B',
          modified: dbFile.modifiedTime ? new Date(dbFile.modifiedTime * 1000).toISOString() : new Date().toISOString(),
          isMediaFile: dbFile.isMediaFile || false,
          isSourceRoot: false,
          processingStatus: dbFile.processingStatus || 'unprocessed',
          tmdbId: dbFile.tmdbId || '',
          seasonNumber: dbFile.seasonNumber || null,
          lastProcessedAt: dbFile.lastProcessedAt || null
        }));

        setSourceFiles(convertedFiles);
        setSourceTotalPages(data.totalPages || 1);
        setSourceTotalFiles(data.total || 0);
        setHasSourceDirectories(true);

        setError('');
      } else {
        setSourceError(`Server error: ${response.status} ${response.statusText}`);
        setSourceFiles([]);
      }

    } catch (err) {
      console.error('Source files fetch error:', err);

      if (axios.isAxiosError(err)) {
        if (err.response) {
          const status = err.response.status;
          const message = err.response.data?.message || err.response.statusText || 'Unknown server error';

          if (status === 503 || (message && message.toLowerCase().includes('source'))) {
            setHasSourceDirectories(false);
          } else {
            setHasSourceDirectories(true);
          }

          setSourceError(`Server error (${status}): ${message}`);
        } else if (err.request) {
          setSourceError('No response from server - check if WebDavHub service is running');
          setHasSourceDirectories(null);
        } else {
          setSourceError(`Request error: ${err.message}`);
          setHasSourceDirectories(null);
        }
      } else {
        setSourceError(`Unexpected error: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setHasSourceDirectories(null);
      }

      setSourceFiles([]);
    } finally {
      setSourceLoading(false);
      setInitialLoading(false);
    }
  }, [sourcePage, sourceIndex]);

  const fetchFileOperations = useCallback(async (showLoading = false) => {
    try {
      if (tabValue === 0) {
        return;
      }

      if (showLoading) {
        setLoading(true);
      }

      const currentSearch = searchQueryRef.current;
      const offset = (currentPage - 1) * recordsPerPage;
      const statusMap = ['created', 'failed', 'skipped', 'deleted'];
      const statusFilter = statusMap[tabValue - 1];

      if (!statusFilter) {
        return;
      }

      const params: any = {
        limit: recordsPerPage,
        offset: offset,
        status: statusFilter,
        lightweight: true,
      };

      if (currentSearch && currentSearch.trim()) {
        params.search = currentSearch.trim();
      }

      const response = await axios.get('/api/file-operations', { params });
      const data = response.data;

      setOperations(data.operations || []);
      setTotalOperations(data.total || 0);
      setStatusCounts(data.statusCounts || {
        created: 0,
        failed: 0,
        skipped: 0,
        deleted: 0,
      });
      setError('');
      setLastUpdated(new Date());
      setInitialLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch file operations');
      setOperations([]);
      setTotalOperations(0);
      setInitialLoading(false);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, [tabValue, currentPage, recordsPerPage]);



  // Initial load effect
  useEffect(() => {
    if (tabValue === 0) {
      fetchSourceFilesData(sourcePage, sourceIndex);
    } else {
      fetchFileOperations(false);
    }
  }, []);

  useEffect(() => {
    const urlCurrentPage = parseInt(searchParams.get('page') || '1', 10);
    const urlSourcePage = parseInt(searchParams.get('sourcePage') || '1', 10);

    if (urlCurrentPage !== currentPageState) {
      setCurrentPageState(urlCurrentPage);
    }
    if (urlSourcePage !== sourcePageState) {
      setSourcePageState(urlSourcePage);
    }
  }, [searchParams, currentPageState, sourcePageState]);

  useEffect(() => {
    if (tabValue === 0) return;

    searchQueryRef.current = searchQuery;
    setIsSearching(true);

    if (currentPage !== 1) {
      setCurrentPage(1);
    } else {
      fetchFileOperations(false);
      setIsSearching(false);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (tabValue === 0) return;
    fetchFileOperations(false);
  }, [currentPage, recordsPerPage]);

  // Separate effect for tab changes to ensure proper data loading
  useEffect(() => {
    if (tabValue > 0) {
      fetchFileOperations(false);
    }
  }, [tabValue]);

  useEffect(() => {
    if (tabValue !== 0) return;
    fetchSourceFilesData(sourcePage, sourceIndex);
  }, [sourcePage, sourceIndex]);

  useEffect(() => {
    if (tabValue !== 0) return;

    sourceSearchQueryRef.current = sourceSearchQuery;
    setIsSourceSearching(true);

    if (sourcePage !== 1) {
      setSourcePage(1);
    } else {
      fetchSourceFilesData(1, sourceIndex);
      setIsSourceSearching(false);
    }
  }, [sourceSearchQuery]);

  const handleOperationUpdate = useCallback((newOperation: FileOperation) => {
    const statusMap = ['created', 'failed', 'skipped', 'deleted'];
    const newOperationTabIndex = statusMap.indexOf(newOperation.status) + 1;

    setStatusCounts(prev => ({
      ...prev,
      [newOperation.status]: prev[newOperation.status] + 1
    }));

    setOperations(prev => {
      const existingIndex = prev.findIndex(op =>
        op.id === newOperation.id ||
        (op.filePath === newOperation.filePath && op.filePath)
      );

      if (existingIndex !== -1) {
        const existingOperation = prev[existingIndex];

        setStatusCounts(prevCounts => ({
          ...prevCounts,
          [existingOperation.status]: Math.max(0, prevCounts[existingOperation.status] - 1)
        }));

        setTotalOperations(prevTotal => Math.max(0, prevTotal - 1));
        return prev.filter((_, index) => index !== existingIndex);
      }

      return prev;
    });

    if (tabValue === newOperationTabIndex) {
      setOperations(prev => [newOperation, ...prev.slice(0, recordsPerPage - 1)]);
      setTotalOperations(prev => prev + 1);
    }

    setLastUpdated(new Date());
  }, [tabValue, recordsPerPage]);

  // Listen for file operation updates through centralized SSE
  useSSEEventListener(
    ['file_operation_update'],
    (event) => {
      const data = event.data;

      if (data && data.operation) {
        const newOperation: FileOperation = {
          id: data.operation.id || `${Date.now()}-${Math.random()}`,
          filePath: data.operation.filePath || '',
          destinationPath: data.operation.destinationPath,
          fileName: data.operation.fileName || data.operation.filePath?.split('/').pop() || 'Unknown',
          status: data.operation.status || 'created',
          timestamp: data.operation.timestamp || new Date().toISOString(),
          reason: data.operation.reason,
          error: data.operation.error,
          tmdbId: data.operation.tmdbId,
          seasonNumber: data.operation.seasonNumber,
          type: data.operation.type || 'other',
          operation: data.operation.operation || 'process'
        };

        handleOperationUpdate(newOperation);
      }
    },
    {
      source: 'file-operations',
      dependencies: [handleOperationUpdate]
    }
  );

  // Listen for file processing events for real-time animations
  useSSEEventListener(
    ['file_processed'],
    (event) => {
      const data = event.data;
      if (data.source_file) {
        setProcessingFiles(prev => new Map(prev.set(data.source_file, {
          fileName: data.filename || data.source_file.split('/').pop() || 'Unknown',
          mediaName: data.media_name,
          mediaType: data.media_type
        })));

        setSourceFiles(prev => prev.map(file => {
          if (file.fullPath === data.source_file) {
            return {
              ...file,
              processingStatus: 'processed',
              tmdbId: data.tmdb_id,
              seasonNumber: data.season_number,
              lastProcessedAt: Date.now()
            };
          }
          return file;
        }));

        // Clear animation after delay but keep the file in the list
        setTimeout(() => {
          setProcessingFiles(prev => {
            const newMap = new Map(prev);
            newMap.delete(data.source_file);
            return newMap;
          });
        }, 3500);
      }
    },
    {
      source: 'mediahub',
      dependencies: []
    }
  );

  useSSEEventListener(
    ['symlink_created'],
    (event) => {
      const data = event.data;
      if (data.source_file) {
        setSourceFiles(prev => prev.map(file => {
          if (file.fullPath === data.source_file) {
            return {
              ...file,
              processingStatus: 'created',
              tmdbId: data.tmdb_id,
              seasonNumber: data.season_number,
              lastProcessedAt: Date.now()
            };
          }
          return file;
        }));

        setProcessingFiles(prev => new Map(prev.set(data.source_file, {
          fileName: data.filename || data.source_file.split('/').pop() || 'Unknown',
          mediaName: data.media_name,
          mediaType: data.media_type
        })));

        setTimeout(() => {
          setProcessingFiles(prev => {
            const newMap = new Map(prev);
            newMap.delete(data.source_file);
            return newMap;
          });
        }, 1500);
      }
    },
    {
      source: 'mediahub',
      dependencies: []
    }
  );

  // Listen for direct source file updates from MediaHub processing
  useSSEEventListener(
    ['source_file_updated'],
    (event) => {
      const data = event.data;
      if (data.file_path) {
        setSourceFiles(prev => prev.map(file => {
          if (file.fullPath === data.file_path) {
            return {
              ...file,
              processingStatus: data.processing_status,
              tmdbId: data.tmdb_id || file.tmdbId,
              seasonNumber: data.season_number || file.seasonNumber,
              lastProcessedAt: data.timestamp || Date.now()
            };
          }
          return file;
        }));
      }
    },
    {
      source: 'mediahub',
      dependencies: []
    }
  );

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setCurrentPage(1);
    setInitialLoading(false);
    setSelectedFiles(new Set());
    setSelectedSourceFiles(new Set());

    if (newValue === 0) {
      fetchSourceFilesData(1, sourceIndex);
      setSourcePage(1);
    }
  };

  const handleMainTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setMainTabValue(newValue);
    setTabValue(0);
  };

  // Source File Browser handlers
  const handleSourceFileClick = (file: FileItem) => {
    if (file.isMediaFile && file.fullPath) {
      if (autoModeEnabled) {
        autoProcessFile(file);
      } else {
        handleProcessFile(file);
      }
    }
  };

  const handleProcessFile = (file: FileItem) => {
    if (file.fullPath) {
      setCurrentFileForProcessing(file.fullPath);
      setModifyDialogOpen(true);
    }
  };

  // Auto-process file with auto-select first result
  const autoProcessFile = async (file: FileItem) => {
    if (!file.fullPath || !file.isMediaFile) return;

    setAutoProcessingFiles(prev => new Set(prev).add(file.fullPath!));

    setProcessingFiles(prev => new Map(prev.set(file.fullPath!, {
      fileName: file.name,
      mediaName: undefined,
      mediaType: undefined
    })));

    try {
      const requestPayload = {
        sourcePath: file.fullPath,
        disableMonitor: true,
        selectedOption: 'auto-select',
        autoSelect: true,
        bulkAutoProcess: false
      };

      const response = await axios.post('/api/python-bridge', requestPayload);

      if (response.status === 200) {
      }
    } catch (error) {
      console.error(`Failed to auto-process file ${file.fullPath}:`, error);

      setProcessingFiles(prev => {
        const newMap = new Map(prev);
        newMap.delete(file.fullPath!);
        return newMap;
      });
    } finally {
      setAutoProcessingFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(file.fullPath!);
        return newSet;
      });
    }
  };

  // Bulk auto-process all unprocessed files from source database
  const bulkAutoProcessFiles = async () => {
    if (bulkProcessing) return;

    setBulkProcessing(true);

    try {
      const requestPayload = {
        sourcePath: '',
        bulkAutoProcess: true,
        disableMonitor: true,
        selectedOption: 'auto-select',
        autoSelect: true,
        batchApply: true
      };

      const response = await axios.post('/api/python-bridge', requestPayload);

      if (response.status === 200) {
        setTimeout(() => {
          if (tabValue === 0) {
            fetchSourceFilesData(sourcePage, sourceIndex);
          }
        }, 2000);
      }
    } catch (error) {
      console.error('Failed to start bulk auto-processing:', error);
    } finally {
      setBulkProcessing(false);
    }
  };

  const handleModifyDialogClose = () => {
    setModifyDialogOpen(false);
    setCurrentFileForProcessing('');
    if (tabValue === 0) {
      fetchSourceFilesData(sourcePage, sourceIndex);
    } else {
      fetchFileOperations(false);
    }
  };

  const handleBulkDeleteSkippedFiles = async () => {
    setBulkDeleteLoading(true);
    try {
      const response = await axios.delete('/api/file-operations');

      if (response.data.success) {
        console.log(response.data.message);

        // Update status counts to reflect the deletion
        setStatusCounts(prev => ({
          ...prev,
          skipped: 0
        }));

        // Clear operations if we're on the skipped tab
        if (tabValue === 3) {
          setOperations([]);
          setTotalOperations(0);
        }

        // Refresh data
        fetchFileOperations(false);

        setBulkDeleteDialogOpen(false);
      }
    } catch (error: any) {
      console.error('Failed to delete skipped files:', error.response?.data?.error || error.message);
      setError(error.response?.data?.error || error.message || 'Failed to delete skipped files');
    } finally {
      setBulkDeleteLoading(false);
    }
  };

  // Bulk selection handlers
  const handleFileSelect = (fileId: string, checked: boolean) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      if (checked) {
        newSet.add(fileId);
      } else {
        newSet.delete(fileId);
      }
      return newSet;
    });
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const allFileIds = operations.map(op => op.id);
      setSelectedFiles(new Set(allFileIds));
    } else {
      setSelectedFiles(new Set());
    }
  };

  const handleBulkDelete = () => {
    setBulkDeleteDialogOpen(true);
  };

  const handleBulkReprocess = () => {
    if (bulkProcessingInProgress) return; // Prevent multiple openings

    console.log('handleBulkReprocess called, opening modify dialog');
    // Get selected operations and their file paths
    const selectedOperations = operations.filter(op => selectedFiles.has(op.id));
    const filePaths = selectedOperations.map(op => op.filePath);

    setBulkModifyFilePaths(filePaths);
    setBulkModifyDialogOpen(true);
    setBulkProcessingInProgress(true);
  };

  // Source file bulk selection handlers
  const handleSourceFileSelect = (filePath: string, checked: boolean) => {
    setSelectedSourceFiles(prev => {
      const newSet = new Set(prev);
      if (checked) {
        newSet.add(filePath);
      } else {
        newSet.delete(filePath);
      }
      return newSet;
    });
  };

  const handleSourceSelectAll = (checked: boolean) => {
    if (checked) {
      const allFilePaths = sourceFiles.map(file => file.fullPath).filter(Boolean) as string[];
      setSelectedSourceFiles(new Set(allFilePaths));
    } else {
      setSelectedSourceFiles(new Set());
    }
  };

  const handleSourceBulkAction = () => {
    if (bulkProcessingInProgress) return; // Prevent multiple openings

    // Get selected source file paths
    const filePaths = Array.from(selectedSourceFiles);

    setBulkModifyFilePaths(filePaths);
    setBulkModifyDialogOpen(true);
    setBulkProcessingInProgress(true);
  };



  const handleBulkDeleteConfirm = async () => {
    if (selectedFiles.size === 0) return;

    setBulkActionLoading(true);
    try {
      // Get selected operations
      const selectedOperations = operations.filter(op => selectedFiles.has(op.id));

      // Check if we're on the deleted tab for permanent deletion
      const statusMap = ['created', 'failed', 'skipped', 'deleted'];
      const statusFilter = statusMap[tabValue - 1];
      const isDeletedTab = statusFilter === 'deleted';

      let totalDeleted = 0;
      let errors: string[] = [];

      if (isDeletedTab) {
        // Permanent deletion for files in deleted tab
        try {
          const filePaths = selectedOperations.map(op => op.id); // Use ID for deleted files
          const response = await axios.delete('/api/file-operations/bulk', {
            data: { filePaths }
          });

          if (response.data.success) {
            totalDeleted = response.data.deletedCount || selectedOperations.length;

            if (response.data.errors && response.data.errors.length > 0) {
              errors = [...errors, ...response.data.errors];
            }
          } else {
            errors.push('Failed to permanently delete selected files');
          }
        } catch (error: any) {
          console.error('Error permanently deleting files:', error);
          errors.push(error.response?.data?.message || 'Failed to permanently delete files');
        }
      } else {
        // Regular deletion logic for other tabs
        // Separate files with destination paths from those without
        const filesWithDestination = selectedOperations.filter(op => op.destinationPath && op.destinationPath.trim() !== '');
        const filesWithoutDestination = selectedOperations.filter(op => !op.destinationPath || op.destinationPath.trim() === '');

        // Delete actual files for operations with destination paths
        if (filesWithDestination.length > 0) {
          try {
            const destinationPaths = filesWithDestination.map(op => op.destinationPath);
            const response = await axios.post('/api/delete', {
              paths: destinationPaths
            });

            if (response.data.success) {
              totalDeleted += response.data.deletedCount || filesWithDestination.length;

              if (response.data.errors && response.data.errors.length > 0) {
                errors = [...errors, ...response.data.errors];
              }
            } else {
              errors.push('Failed to delete files');
            }
          } catch (error: any) {
            console.error('Failed to delete files with destination paths:', error);
            errors.push(`Failed to delete ${filesWithDestination.length} files: ${error.response?.data?.error || error.message}`);
          }
        }

        // Delete database records for files without destination paths (failed files)
        if (filesWithoutDestination.length > 0) {
          try {
            const filePaths = filesWithoutDestination.map(op => op.filePath);
            const response = await axios.delete('/api/file-operations/bulk', {
              data: { filePaths }
            });

            if (response.data.success) {
              totalDeleted += response.data.deletedCount || filesWithoutDestination.length;
            }
          } catch (error: any) {
            console.error('Failed to delete database records:', error);
            errors.push(`Failed to delete ${filesWithoutDestination.length} database records: ${error.response?.data?.error || error.message}`);
          }
        }
      }

      if (totalDeleted > 0) {
        console.log(`Bulk deleted ${totalDeleted} files/records`);

        if (errors.length > 0) {
          console.warn('Some files could not be deleted:', errors);
          setError(`Deleted ${totalDeleted} items, but ${errors.length} errors occurred. Check console for details.`);
        }

        // Remove deleted files from operations
        setOperations(prev => prev.filter(op => !selectedFiles.has(op.id)));
        setTotalOperations(prev => prev - totalDeleted);

        // Update status counts
        const deletedByStatus = selectedOperations.reduce((acc, op) => {
          acc[op.status] = (acc[op.status] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);

        setStatusCounts(prev => {
          const updated = { ...prev };
          Object.entries(deletedByStatus).forEach(([status, count]) => {
            if (status in updated) {
              (updated as any)[status] = Math.max(0, (updated as any)[status] - count);
            }
          });
          return updated;
        });
      } else if (errors.length > 0) {
        setError(`Failed to delete any files. ${errors.length} errors occurred.`);
      }

      setSelectedFiles(new Set());
      setBulkDeleteDialogOpen(false);
    } catch (error: any) {
      console.error('Failed to delete selected files:', error.response?.data?.error || error.message);
      setError(error.response?.data?.error || error.message || 'Failed to delete selected files');
    } finally {
      setBulkActionLoading(false);
    }
  };

  const handleBulkModifyDialogClose = () => {
    // Clean up bulk processing
    setBulkModifyDialogOpen(false);
    setBulkModifyFilePaths([]);
    setSelectedFiles(new Set());
    setSelectedSourceFiles(new Set());
    setBulkProcessingInProgress(false);

    // Refresh data based on current tab
    if (tabValue === 0) {
      fetchSourceFilesData(sourcePage, sourceIndex);
    } else {
      fetchFileOperations(false);
    }
  };

  const getProcessingStatus = (file: any): any => {
    if (!file.isMediaFile) return null;

    if (file.processingStatus && file.processingStatus !== 'unprocessed') {
      return {
        status: file.processingStatus,
        tmdbId: file.tmdbId,
        seasonNumber: file.seasonNumber,
        lastProcessedAt: file.lastProcessedAt
      };
    }

    return null;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'created':
        return theme.palette.success.main;
      case 'failed':
        return theme.palette.error.main;
      case 'deleted':
        return theme.palette.info.main;
      case 'skipped':
        return theme.palette.secondary.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'created':
        return <CheckCircle sx={{ fontSize: 16, color: getStatusColor(status) }} />;
      case 'failed':
        return <WarningIcon sx={{ fontSize: 16, color: getStatusColor(status) }} />;
      case 'deleted':
        return <DeleteIcon sx={{ fontSize: 16, color: getStatusColor(status) }} />;
      case 'skipped':
        return <SkipIcon sx={{ fontSize: 16, color: getStatusColor(status) }} />;
      default:
        return <CheckCircle sx={{ fontSize: 16, color: getStatusColor(status) }} />;
    }
  };

  // Source File Browser helper functions
  const getFileIcon = (file: FileItem) => {
    if (file.isSourceRoot) {
      return <DatabaseIcon sx={{ color: 'primary.main', fontSize: 28 }} />;
    }
    if (file.type === 'directory') {
      return <FolderOpenIcon sx={{ color: 'warning.main', fontSize: 24 }} />;
    }
    if (file.isMediaFile) {
      const status = getProcessingStatus(file);
      const fileName = file.name.toLowerCase();
      const isProcessed = status?.status === 'processed' || status?.status === 'created';

      const iconStyle = {
        fontSize: 24,
        color: isProcessed ? 'success.main' : 'text.secondary',
        opacity: isProcessed ? 1 : 0.7,
        position: 'relative' as const,
      };

      if (fileName.includes('s0') || fileName.includes('season') || fileName.includes('episode') || fileName.includes('e0')) {
        return (
          <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <TvIcon sx={iconStyle} />
            {isProcessed && (
              <LinkIcon sx={{
                position: 'absolute',
                top: -2,
                right: -2,
                fontSize: 12,
                color: 'success.main',
                bgcolor: 'background.paper',
                borderRadius: '50%',
                p: 0.2
              }} />
            )}
          </Box>
        );
      } else {
        return (
          <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <MovieIcon sx={iconStyle} />
            {isProcessed && (
              <LinkIcon sx={{
                position: 'absolute',
                top: -2,
                right: -2,
                fontSize: 12,
                color: 'success.main',
                bgcolor: 'background.paper',
                borderRadius: '50%',
                p: 0.2
              }} />
            )}
          </Box>
        );
      }
    }
    return <FileIcon sx={{ color: 'text.secondary', fontSize: 20 }} />;
  };

  const getFileTypeChip = (file: FileItem) => {
    if (file.isSourceRoot) {
      return (
        <Chip
          label="Source Directory"
          size="small"
          color="primary"
          variant="outlined"
          icon={<DatabaseIcon />}
        />
      );
    }
    return null;
  };

  const getStatusTooltip = (file: any): string => {
    const status = getProcessingStatus(file);
    if (!status) {
      return file.isMediaFile ? 'This file has not been processed yet' : '';
    }

    let tooltip = `Status: ${status.status.toUpperCase()}`;

    if (status.lastProcessedAt) {
      const timestamp = new Date(status.lastProcessedAt * 1000).toLocaleString();
      tooltip += `\nProcessed: ${timestamp}`;
    }

    if (status.tmdbId) {
      tooltip += `\nTMDB ID: ${status.tmdbId}`;
    }

    if (status.seasonNumber) {
      tooltip += `\nSeason: ${status.seasonNumber}`;
    }

    return tooltip;
  };

  const getProcessingStatusChip = (file: FileItem) => {
    const status = getProcessingStatus(file);

    if (!status) {
      if (file.isMediaFile) {
        return (
          <Chip
            label="Not Processed"
            size="small"
            color="default"
            variant="outlined"
            icon={<UnprocessedIcon sx={{ fontSize: 16 }} />}
            sx={{
              bgcolor: alpha(theme.palette.grey[500], 0.1),
              color: 'text.secondary',
              fontWeight: 500,
              border: `1px solid ${alpha(theme.palette.grey[500], 0.3)}`,
              '& .MuiChip-icon': {
                color: 'text.secondary'
              }
            }}
          />
        );
      }
      return null;
    }

    switch (status.status) {
      case 'processed':
      case 'created':
        return (
          <Chip
            label="Processed"
            size="small"
            color="success"
            variant="filled"
            icon={<ProcessedIcon sx={{ fontSize: 16 }} />}
            sx={{
              bgcolor: alpha(theme.palette.success.main, 0.15),
              color: 'success.main',
              fontWeight: 600,
              border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
              '& .MuiChip-icon': {
                color: 'success.main'
              }
            }}
          />
        );

      case 'failed':
        return (
          <Chip
            label="Failed"
            size="small"
            color="error"
            variant="filled"
            icon={<WarningIcon2 sx={{ fontSize: 16 }} />}
            sx={{
              bgcolor: alpha(theme.palette.error.main, 0.15),
              color: 'error.main',
              fontWeight: 600,
              border: `1px solid ${alpha(theme.palette.error.main, 0.3)}`,
              '& .MuiChip-icon': {
                color: 'error.main'
              }
            }}
          />
        );

      case 'skipped':
        return (
          <Chip
            label="Skipped"
            size="small"
            color="warning"
            variant="filled"
            icon={<WarningIcon2 sx={{ fontSize: 16 }} />}
            sx={{
              bgcolor: alpha(theme.palette.warning.main, 0.15),
              color: 'warning.main',
              fontWeight: 600,
              border: `1px solid ${alpha(theme.palette.warning.main, 0.3)}`,
              '& .MuiChip-icon': {
                color: 'warning.main'
              }
            }}
          />
        );

      case 'deleted':
        return (
          <Chip
            label="Deleted"
            size="small"
            color="info"
            variant="filled"
            icon={<DeleteIcon sx={{ fontSize: 16 }} />}
            sx={{
              bgcolor: alpha(theme.palette.info.main, 0.15),
              color: 'info.main',
              fontWeight: 600,
              border: `1px solid ${alpha(theme.palette.info.main, 0.3)}`,
              '& .MuiChip-icon': {
                color: 'info.main'
              }
            }}
          />
        );

      default:
        return null;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    if (isMobile) {
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleString();
  };

  const toggleCardExpansion = (cardId: string) => {
    setExpandedCards(prev => {
      const newSet = new Set(prev);
      if (newSet.has(cardId)) {
        newSet.delete(cardId);
      } else {
        newSet.add(cardId);
      }
      return newSet;
    });
  };



  const renderMobileCard = React.useCallback((file: FileOperation, _index: number) => {
    const isExpanded = expandedCards.has(file.id);
    const isSelected = selectedFiles.has(file.id);

    return (
      <Card
        key={file.id}
        sx={{
          mb: 1.5,
          borderRadius: 3,
          border: '1px solid',
          borderColor: isSelected ? 'primary.main' : 'divider',
          bgcolor: isSelected ? alpha(theme.palette.primary.main, 0.05) : 'background.paper',
          overflow: 'hidden',
          transition: 'all 0.2s ease',
        }}
      >
        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              mb: isExpanded ? 1.5 : 0,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, minWidth: 0, flex: 1 }}>
              <Checkbox
                checked={isSelected}
                onChange={(e) => {
                  e.stopPropagation();
                  handleFileSelect(file.id, e.target.checked);
                }}
                size="small"
                sx={{
                  p: 0.5,
                  color: 'text.secondary',
                  '&.Mui-checked': {
                    color: 'primary.main',
                  },
                }}
              />
              <Box sx={{
                p: 1,
                borderRadius: 2,
                bgcolor: alpha(getStatusColor(file.status), 0.1),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                {getStatusIcon(file.status)}
              </Box>
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <Typography variant="body2" fontWeight="600" sx={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  color: 'text.primary',
                }}>
                  {file.fileName}
                </Typography>
                <Box sx={{
                  display: 'flex',
                  alignItems: isMobile ? 'flex-start' : 'center',
                  gap: 1,
                  mt: 0.5,
                  flexDirection: isMobile ? 'column' : 'row'
                }}>
                  <Chip
                    label={file.status.toUpperCase()}
                    size="small"
                    sx={{
                      height: 20,
                      fontSize: '0.7rem',
                      fontWeight: 600,
                      bgcolor: alpha(getStatusColor(file.status), 0.1),
                      color: getStatusColor(file.status),
                      border: 'none',
                    }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5,
                    mt: isMobile ? 0.5 : 0,
                  }}>
                    <ScheduleIcon sx={{ fontSize: 12 }} />
                    {formatTimestamp(file.timestamp)}
                  </Typography>
                </Box>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {/* Action buttons for created, skipped, and failed files */}
              {(file.status === 'created' || file.status === 'skipped' || file.status === 'failed') && (
                <Tooltip title={
                  file.status === 'created' ? 'File Actions' :
                    file.status === 'failed' ? 'Retry Processing' :
                      'Reprocess File'
                }>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      // Open the modify dialog for file processing
                      setCurrentFileForProcessing(file.filePath || '');
                      setModifyDialogOpen(true);
                    }}
                    sx={{
                      bgcolor: alpha(
                        file.status === 'created'
                          ? theme.palette.success.main
                          : file.status === 'failed'
                            ? theme.palette.error.main
                            : theme.palette.warning.main,
                        0.1
                      ),
                      color: file.status === 'created' ? 'success.main' :
                        file.status === 'failed' ? 'error.main' : 'warning.main',
                      '&:hover': {
                        bgcolor: alpha(
                          file.status === 'created'
                            ? theme.palette.success.main
                            : file.status === 'failed'
                              ? theme.palette.error.main
                              : theme.palette.warning.main,
                          0.2
                        ),
                        transform: 'scale(1.1)',
                      },
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <SettingsIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
              )}

              {/* Restore button for deleted files */}
              {file.status === 'deleted' && (
                <Tooltip title="Restore File">
                  <IconButton
                    size="small"
                    onClick={async (e) => {
                      e.stopPropagation();
                      try {
                        setBulkActionLoading(true);
                        const path = file.destinationPath || file.filePath;

                        if (!path) {
                          console.error('No path available for restoration');
                          return;
                        }

                        console.log('Restoring single file:', path);

                        const response = await axios.post('/api/restore-symlinks', {
                          paths: [path]
                        });

                        if (response.data.success) {
                          setError('');
                          setSuccessMessage(`File "${file.fileName}" restored successfully!`);
                          setSuccessDialogOpen(true);

                          setOperations(prev => prev.filter(op => op.id !== file.id));
                          setTotalOperations(prev => prev - 1);

                          setStatusCounts(prev => ({
                            ...prev,
                            deleted: prev.deleted - 1
                          }));

                          setTimeout(async () => {
                            await fetchFileOperations(false);
                          }, 1000);
                        } else {
                          setSuccessMessage('');
                          setError(`Failed to restore "${file.fileName}": ${response.data.message || 'Unknown error'}`);
                        }
                      } catch (error: any) {
                        setSuccessMessage('');
                        setError(`Error restoring "${file.fileName}": ${error.response?.data?.message || error.message || 'Network error'}`);
                      } finally {
                        setBulkActionLoading(false);
                      }
                    }}
                    disabled={bulkActionLoading}
                    sx={{
                      bgcolor: alpha(theme.palette.success.main, 0.1),
                      color: 'success.main',
                      '&:hover': {
                        bgcolor: alpha(theme.palette.success.main, 0.2),
                        transform: 'scale(1.1)',
                      },
                      '&:disabled': {
                        bgcolor: alpha(theme.palette.success.main, 0.05),
                        color: alpha(theme.palette.success.main, 0.5),
                      },
                      transition: 'all 0.2s ease'
                    }}
                  >
                    {bulkActionLoading ?
                      <CircularProgress size={18} sx={{ color: 'success.main' }} /> :
                      <RestoreIcon sx={{ fontSize: 18 }} />
                    }
                  </IconButton>
                </Tooltip>
              )}

              <IconButton
                size="small"
                onClick={() => toggleCardExpansion(file.id)}
                sx={{ color: 'text.secondary' }}
              >
                {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Box>
          </Box>

          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                style={{ overflow: 'hidden' }}
              >
                <Divider sx={{ mb: 1.5 }} />
                <Stack spacing={1.5}>
                  <Box>
                    <Typography variant="caption" color="text.secondary" fontWeight="600">
                      Source Path
                    </Typography>
                    <Typography variant="body2" sx={{
                      wordBreak: 'break-all',
                      bgcolor: 'action.hover',
                      p: 1,
                      borderRadius: 1,
                      mt: 0.5,
                      fontSize: '0.8rem',
                    }}>
                      {file.filePath}
                    </Typography>
                  </Box>

                  {file.destinationPath && (
                    <Box>
                      <Typography variant="caption" color="text.secondary" fontWeight="600">
                        Destination Path
                      </Typography>
                      <Typography variant="body2" sx={{
                        wordBreak: 'break-all',
                        bgcolor: 'action.hover',
                        p: 1,
                        borderRadius: 1,
                        mt: 0.5,
                        fontSize: '0.8rem',
                      }}>
                        {file.destinationPath}
                      </Typography>
                    </Box>
                  )}

                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={file.type.toUpperCase()}
                      size="small"
                      sx={{
                        bgcolor: alpha(theme.palette.info.main, 0.1),
                        color: 'info.main',
                        fontWeight: 600,
                        fontSize: '0.7rem',
                      }}
                    />
                    <Chip
                      label={file.operation === 'delete' ? 'DELETE' : 'PROCESS'}
                      size="small"
                      sx={{
                        bgcolor: file.operation === 'delete'
                          ? alpha(theme.palette.error.main, 0.1)
                          : alpha(theme.palette.success.main, 0.1),
                        color: file.operation === 'delete' ? 'error.main' : 'success.main',
                        fontWeight: 600,
                        fontSize: '0.7rem',
                      }}
                    />
                    {file.seasonNumber && (
                      <Chip
                        label={`Season ${file.seasonNumber}`}
                        size="small"
                        sx={{
                          bgcolor: alpha(theme.palette.secondary.main, 0.1),
                          color: 'secondary.main',
                          fontWeight: 600,
                          fontSize: '0.7rem',
                        }}
                      />
                    )}
                  </Box>

                  {(file.reason || file.error) && (
                    <Box>
                      <Typography variant="caption" color="text.secondary" fontWeight="600">
                        {file.error ? 'Error' : 'Reason'}
                      </Typography>
                      <Typography variant="body2" color={file.error ? 'error.main' : 'warning.main'} sx={{
                        bgcolor: file.error
                          ? alpha(theme.palette.error.main, 0.1)
                          : alpha(theme.palette.warning.main, 0.1),
                        p: 1,
                        borderRadius: 1,
                        mt: 0.5,
                        fontSize: '0.8rem',
                      }}>
                        {file.error || file.reason}
                      </Typography>
                    </Box>
                  )}
                </Stack>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    );
  }, [expandedCards, theme, selectedFiles]);

  const renderFileTable = (files: FileOperation[], emptyMessage: string) => {
    if (files.length === 0) {
      return (
        <Box
          sx={{
            textAlign: 'center',
            py: { xs: 6, sm: 8 },
            bgcolor: 'background.paper',
            borderRadius: 3,
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography variant="h6" color="text.secondary" sx={{
            mb: 1,
            fontSize: { xs: '1rem', sm: '1.25rem' }
          }}>
            {emptyMessage}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{
            fontSize: { xs: '0.8rem', sm: '0.875rem' }
          }}>
            File operations will appear here as they are processed.
          </Typography>
        </Box>
      );
    }

    // Card layout (preferred for all screen sizes)
    return (
      <Box sx={{ px: { xs: 0, sm: 1, md: 2 } }}>
        {files.length > 20 ? (
          // Disable animations for large lists to improve performance
          (files.map((file, index) => renderMobileCard(file, index)))
        ) : (
          <AnimatePresence>
            {files.map((file, index) => renderMobileCard(file, index))}
          </AnimatePresence>
        )}
      </Box>
    );
  };

  if (initialLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
          gap: 2
        }}
      >
        <CircularProgress size={40} />
        <Typography variant="h6" color="text.secondary">
          Loading file operations...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{
      px: { xs: 1, sm: 1, md: 0 },
      maxWidth: 1400,
      mx: 'auto',
      pb: { xs: 10, sm: 4 }, // Extra bottom padding for mobile FAB
      position: 'relative'
    }}>
      {/* Header */}
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        mb: { xs: 2, sm: 2.5 },
        py: { xs: 1, sm: 0 },
      }}>
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          gap: { xs: 1, sm: 1.5 }
        }}>
          <Box sx={{
            backgroundColor: `${theme.palette.primary.main}15`,
            borderRadius: { xs: '10px', sm: '12px' },
            p: { xs: 0.6, sm: 0.8 },
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: `1px solid ${theme.palette.primary.main}30`,
          }}>
            <AssignmentIcon sx={{
              color: 'primary.main',
              fontSize: { xs: 16, sm: 20, md: 22 }
            }} />
          </Box>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 700,
              letterSpacing: 0.3,
              fontSize: { xs: '1rem', sm: '1.3rem', md: '1.75rem' }
            }}
          >
            File Operations
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {lastUpdated && (
            <Typography variant="caption" color="text.secondary" sx={{
              display: { xs: 'none', sm: 'block' },
              fontSize: { sm: '0.7rem', md: '0.75rem' }
            }}>
              Updated: {lastUpdated.toLocaleTimeString()}
            </Typography>
          )}


        </Box>
      </Box>
      {error && (
        <Alert
          severity="error"
          onClose={() => setError('')}
          sx={{ mb: 3 }}
        >
          {error}
        </Alert>
      )}

      {/* Main Tab Navigation */}
      <Box sx={{ mb: 4 }}>
        <Box
          sx={{
            display: 'flex',
            gap: 0.5,
            p: 0.5,
            bgcolor: alpha(theme.palette.background.paper, 0.8),
            borderRadius: 3,
            border: '1px solid',
            borderColor: alpha(theme.palette.divider, 0.5),
            width: { xs: '100%', sm: 'fit-content' },
            maxWidth: '100%',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            overflowX: { xs: 'auto', sm: 'visible' },
            '&::-webkit-scrollbar': {
              display: 'none',
            },
            scrollbarWidth: 'none',
          }}
        >
          {[
            { name: 'Operations', icon: <OperationsIcon />, color: '#3b82f6' },
            { name: 'Database', icon: <DatabaseIcon />, color: '#4CAF50' }
          ].map((tab, index) => {
            const isSelected = mainTabValue === index;
            return (
              <Box
                key={tab.name}
                onClick={() => handleMainTabChange({} as React.SyntheticEvent, index)}
                sx={{
                  cursor: 'pointer',
                  px: { xs: 2.5, sm: 4 },
                  py: { xs: 1.5, sm: 2 },
                  borderRadius: 2.5,
                  bgcolor: isSelected
                    ? `linear-gradient(135deg, ${tab.color} 0%, ${alpha(tab.color, 0.8)} 100%)`
                    : 'transparent',
                  background: isSelected
                    ? `linear-gradient(135deg, ${tab.color} 0%, ${alpha(tab.color, 0.8)} 100%)`
                    : 'transparent',
                  color: isSelected ? 'white' : 'text.primary',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  flex: { xs: '1 1 0', sm: '0 0 auto' },
                  minWidth: { xs: 0, sm: 'auto' },
                  whiteSpace: 'nowrap',
                  position: 'relative',
                  overflow: 'hidden',
                  '&:hover': {
                    bgcolor: isSelected
                      ? `linear-gradient(135deg, ${tab.color} 0%, ${alpha(tab.color, 0.8)} 100%)`
                      : alpha(tab.color, 0.1),
                    background: isSelected
                      ? `linear-gradient(135deg, ${tab.color} 0%, ${alpha(tab.color, 0.8)} 100%)`
                      : alpha(tab.color, 0.1),
                    transform: 'translateY(-1px)',
                    boxShadow: isSelected
                      ? `0 12px 24px ${alpha(tab.color, 0.3)}`
                      : `0 4px 12px ${alpha(tab.color, 0.2)}`,
                  },
                  '&:active': {
                    transform: 'translateY(0px)',
                  },
                  '&::before': isSelected ? {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: `linear-gradient(135deg, ${alpha('#ffffff', 0.1)} 0%, transparent 50%)`,
                    borderRadius: 'inherit',
                    pointerEvents: 'none',
                  } : {},
                }}
              >
                <Stack direction="row" alignItems="center" spacing={{ xs: 1, sm: 1.5 }} sx={{ minWidth: 0, justifyContent: 'center' }}>
                  <Box
                    sx={{
                      width: { xs: 20, sm: 24 },
                      height: { xs: 20, sm: 24 },
                      borderRadius: 1,
                      bgcolor: isSelected ? 'rgba(255, 255, 255, 0.2)' : alpha(tab.color, 0.15),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: isSelected ? 'white' : tab.color,
                      transition: 'all 0.3s ease',
                      flexShrink: 0,
                      '& svg': {
                        fontSize: { xs: 16, sm: 18 },
                      },
                    }}
                  >
                    {tab.icon}
                  </Box>
                  <Typography
                    variant="body1"
                    fontWeight="600"
                    sx={{
                      fontSize: { xs: '0.9rem', sm: '1rem' },
                      letterSpacing: '0.02em',
                      flexShrink: 0,
                      minWidth: 0,
                      textAlign: 'center',
                    }}
                  >
                    {tab.name}
                  </Typography>
                </Stack>
              </Box>
            );
          })}
        </Box>
      </Box>
      {/* Operations Tab Content */}
      {mainTabValue === 0 && (
        <>
          {/* Sub-Tabs for Operations */}
          <Box sx={{
            borderBottom: 1,
            borderColor: 'divider',
            mb: { xs: 2, sm: 3 },
            mx: { xs: -1, sm: 0 },
            px: { xs: 1, sm: 0 },
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              aria-label="file operations tabs"
              variant={isMobile ? "scrollable" : "standard"}
              scrollButtons={isMobile ? "auto" : false}
              allowScrollButtonsMobile={isMobile}
              sx={{
                flex: 1,
                '& .MuiTab-root': {
                  textTransform: 'none',
                  fontWeight: 600,
                  fontSize: { xs: '0.7rem', sm: '0.875rem' },
                  minHeight: { xs: 40, sm: 48 },
                  minWidth: { xs: 60, sm: 160 },
                  px: { xs: 0.5, sm: 2 },
                },
                '& .MuiTabs-scrollButtons': {
                  color: 'primary.main',
                },
                '& .MuiTabs-flexContainer': {
                  gap: { xs: 0, sm: 1 },
                },
              }}
            >
              <Tab
                label={
                  <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: { xs: 0.5, sm: 1 },
                    flexDirection: { xs: 'column', sm: 'row' }
                  }}>
                    <SourceIcon sx={{ fontSize: { xs: 16, sm: 18 } }} />
                    <Typography variant="caption" sx={{
                      fontSize: { xs: '0.65rem', sm: '0.75rem' },
                      display: { xs: 'block', sm: 'inline' },
                      lineHeight: 1.2,
                    }}>
                      Source
                    </Typography>
                  </Box>
                }
                {...a11yProps(0)}
              />
              <Tab
                label={
                  <Badge badgeContent={statusCounts.created} color="success" max={999}>
                    <Box sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: { xs: 0.5, sm: 1 },
                      flexDirection: { xs: 'column', sm: 'row' }
                    }}>
                      <CheckCircle sx={{ fontSize: { xs: 16, sm: 18 } }} />
                      <Typography variant="caption" sx={{
                        fontSize: { xs: '0.65rem', sm: '0.75rem' },
                        display: { xs: 'block', sm: 'inline' },
                        lineHeight: 1.2,
                      }}>
                        Created
                      </Typography>
                    </Box>
                  </Badge>
                }
                {...a11yProps(1)}
              />
              <Tab
                label={
                  <Badge badgeContent={statusCounts.failed} color="error" max={999}>
                    <Box sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: { xs: 0.5, sm: 1 },
                      flexDirection: { xs: 'column', sm: 'row' }
                    }}>
                      <WarningIcon sx={{ fontSize: { xs: 16, sm: 18 } }} />
                      <Typography variant="caption" sx={{
                        fontSize: { xs: '0.65rem', sm: '0.75rem' },
                        display: { xs: 'block', sm: 'inline' },
                        lineHeight: 1.2,
                      }}>
                        Failed
                      </Typography>
                    </Box>
                  </Badge>
                }
                {...a11yProps(2)}
              />
              <Tab
                label={
                  <Badge badgeContent={statusCounts.skipped} color="secondary" max={999}>
                    <Box sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: { xs: 0.5, sm: 1 },
                      flexDirection: { xs: 'column', sm: 'row' }
                    }}>
                      <SkipIcon sx={{ fontSize: { xs: 16, sm: 18 } }} />
                      <Typography variant="caption" sx={{
                        fontSize: { xs: '0.65rem', sm: '0.75rem' },
                        display: { xs: 'block', sm: 'inline' },
                        lineHeight: 1.2,
                      }}>
                        Skipped
                      </Typography>
                    </Box>
                  </Badge>
                }
                {...a11yProps(3)}
              />
              <Tab
                label={
                  <Badge badgeContent={statusCounts.deleted} color="info" max={999}>
                    <Box sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: { xs: 0.5, sm: 1 },
                      flexDirection: { xs: 'column', sm: 'row' }
                    }}>
                      <DeleteIcon sx={{ fontSize: { xs: 16, sm: 18 } }} />
                      <Typography variant="caption" sx={{
                        fontSize: { xs: '0.65rem', sm: '0.75rem' },
                        display: { xs: 'block', sm: 'inline' },
                        lineHeight: 1.2,
                      }}>
                        Deleted
                      </Typography>
                    </Box>
                  </Badge>
                }
                {...a11yProps(4)}
              />
            </Tabs>

            {/* Bulk Delete Button for Skipped Files - Tabs Line Placement */}
            {!isMobile && tabValue === 3 && statusCounts.skipped > 0 && (
              <Tooltip title={`Delete all ${statusCounts.skipped} skipped files from database`}>
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  startIcon={<DeleteSweepIcon />}
                  onClick={() => setBulkDeleteDialogOpen(true)}
                  disabled={bulkDeleteLoading}
                  sx={{
                    borderRadius: 2,
                    px: 2,
                    py: 0.5,
                    fontWeight: 600,
                    textTransform: 'none',
                    fontSize: '0.75rem',
                    minWidth: 'auto',
                    border: '1px solid',
                    borderColor: 'error.main',
                    color: 'error.main',
                    bgcolor: alpha(theme.palette.error.main, 0.05),
                    ml: 2,
                    '&:hover': {
                      bgcolor: alpha(theme.palette.error.main, 0.1),
                      borderColor: 'error.dark',
                      transform: 'translateY(-1px)',
                      boxShadow: `0 2px 8px ${alpha(theme.palette.error.main, 0.25)}`,
                    },
                    '&:disabled': {
                      bgcolor: alpha(theme.palette.error.main, 0.02),
                      borderColor: alpha(theme.palette.error.main, 0.3),
                      color: alpha(theme.palette.error.main, 0.5),
                    },
                  }}
                >
                  {bulkDeleteLoading ? (
                    <>
                      <CircularProgress size={14} sx={{ mr: 1 }} />
                      Deleting...
                    </>
                  ) : (
                    `Delete All ${statusCounts.skipped}`
                  )}
                </Button>
              </Tooltip>
            )}
          </Box>

          {/* Bulk Selection Toolbar */}
          {(tabValue === 0 ? selectedSourceFiles.size > 0 : selectedFiles.size > 0) && (
            <Collapse in={tabValue === 0 ? selectedSourceFiles.size > 0 : selectedFiles.size > 0} timeout={300}>
              <Box sx={{ mb: 3 }}>
                <Box
                  sx={{
                    background: theme.palette.mode === 'dark'
                      ? `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.15)} 0%, ${alpha(theme.palette.primary.main, 0.08)} 100%)`
                      : `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.08)} 0%, ${alpha(theme.palette.primary.main, 0.04)} 100%)`,
                    backdropFilter: 'blur(20px)',
                    borderRadius: 4,
                    border: theme.palette.mode === 'dark'
                      ? `1px solid ${alpha(theme.palette.primary.main, 0.3)}`
                      : `1px solid ${alpha(theme.palette.primary.main, 0.15)}`,
                    boxShadow: theme.palette.mode === 'dark'
                      ? `0 8px 32px ${alpha(theme.palette.primary.main, 0.15)}, 0 2px 8px ${alpha('#000', 0.3)}`
                      : `0 8px 32px ${alpha(theme.palette.primary.main, 0.12)}, 0 2px 8px ${alpha('#000', 0.08)}`,
                    px: 3,
                    py: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    position: 'relative',
                    overflow: 'hidden',
                    opacity: selectedFiles.size > 0 ? 1 : 0,
                    transform: selectedFiles.size > 0 ? 'scale(1)' : 'scale(0.95)',
                    transition: 'opacity 0.2s ease-out, transform 0.2s ease-out',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      height: '2px',
                      background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      borderRadius: '4px 4px 0 0',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1.5,
                        px: 2,
                        py: 1,
                        borderRadius: 3,
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                      }}
                    >
                      <Checkbox
                        checked={tabValue === 0
                          ? selectedSourceFiles.size === sourceFiles.filter(f => f.isMediaFile).length && sourceFiles.filter(f => f.isMediaFile).length > 0
                          : selectedFiles.size === operations.length && operations.length > 0
                        }
                        indeterminate={tabValue === 0
                          ? selectedSourceFiles.size > 0 && selectedSourceFiles.size < sourceFiles.filter(f => f.isMediaFile).length
                          : selectedFiles.size > 0 && selectedFiles.size < operations.length
                        }
                        onChange={(e) => tabValue === 0 ? handleSourceSelectAll(e.target.checked) : handleSelectAll(e.target.checked)}
                        size="small"
                        sx={{
                          p: 0,
                          color: theme.palette.primary.main,
                          '&.Mui-checked': {
                            color: theme.palette.primary.main,
                          },
                          '&.MuiCheckbox-indeterminate': {
                            color: theme.palette.primary.main,
                          },
                        }}
                      />
                      <Typography
                        variant="body2"
                        fontWeight="600"
                        sx={{
                          color: theme.palette.primary.main,
                          fontSize: '0.875rem',
                        }}
                      >
                        {tabValue === 0 ? (
                          selectedSourceFiles.size === sourceFiles.filter(f => f.isMediaFile).length && sourceFiles.filter(f => f.isMediaFile).length > 0
                            ? `All ${sourceFiles.filter(f => f.isMediaFile).length} selected`
                            : selectedSourceFiles.size > 0
                              ? `${selectedSourceFiles.size} selected`
                              : 'Select all'
                        ) : (
                          selectedFiles.size === operations.length && operations.length > 0
                            ? `All ${operations.length} selected`
                            : selectedFiles.size > 0
                              ? `${selectedFiles.size} selected`
                              : 'Select all'
                        )}
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 1.5 }}>
                    {tabValue === 0 ? (
                      // Source Files Tab - Reprocess Action
                      <Button
                        variant="contained"
                        size="medium"
                        startIcon={<PlayArrowIcon />}
                        onClick={() => handleSourceBulkAction()}
                        disabled={selectedSourceFiles.size === 0 || bulkActionLoading}
                        sx={{
                          bgcolor: theme.palette.primary.main,
                          color: '#fff',
                          fontWeight: 600,
                          borderRadius: 3,
                          px: 3,
                          py: 1,
                          textTransform: 'none',
                          boxShadow: `0 4px 16px ${alpha(theme.palette.primary.main, 0.3)}`,
                          border: 'none',
                          '&:hover': {
                            bgcolor: theme.palette.primary.dark,
                            boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                            transform: 'translateY(-1px)',
                          },
                          '&:active': {
                            transform: 'translateY(0)',
                          },
                          '&:disabled': {
                            bgcolor: alpha(theme.palette.primary.main, 0.3),
                            color: alpha('#fff', 0.5),
                            boxShadow: 'none',
                          },
                          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                        }}
                      >
                        {bulkActionLoading ? 'Processing...' : 'Reprocess Selected'}
                      </Button>
                    ) : tabValue === 4 ? (
                      // Deleted Tab - Delete and Restore Actions
                      <>
                        <Button
                          variant="outlined"
                          size="medium"
                          startIcon={<RestoreIcon />}
                          onClick={async () => {
                            try {
                              setBulkActionLoading(true);
                              const selectedOps = operations.filter(op => selectedFiles.has(op.id));
                              const paths = selectedOps.map(op => op.destinationPath || op.filePath).filter(Boolean);

                              if (paths.length === 0) {
                                setSuccessMessage('');
                                setError('No valid paths found for restoration');
                                return;
                              }

                              const response = await axios.post('/api/restore-symlinks', { paths });

                              if (response.data.success) {
                                setError('');
                                const count = response.data.restoredCount;
                                setSuccessMessage(`Successfully restored ${count} file${count !== 1 ? 's' : ''}!`);
                                setSuccessDialogOpen(true);
                                setSelectedFiles(new Set());
                                setTimeout(async () => {
                                  await fetchFileOperations(false);
                                }, 1000);
                              } else {
                                setSuccessMessage('');
                                const errorMsg = response.data.errors?.join(', ') || 'Unknown error';
                                setError(`Failed to restore files: ${errorMsg}`);
                              }
                            } catch (e) {
                              setSuccessMessage('');
                              if (axios.isAxiosError(e)) {
                                setError(`Restore failed: ${e.response?.data?.message || e.message}`);
                              } else {
                                setError(`Restore failed: ${String(e)}`);
                              }
                            } finally {
                              setBulkActionLoading(false);
                            }
                          }}
                          disabled={selectedFiles.size === 0 || bulkActionLoading}
                          sx={{
                            borderColor: theme.palette.success.main,
                            color: theme.palette.success.main,
                            fontWeight: 600,
                            borderRadius: 3,
                            px: 3,
                            py: 1,
                            textTransform: 'none',
                            '&:hover': {
                              bgcolor: alpha(theme.palette.success.main, 0.1),
                              borderColor: theme.palette.success.dark,
                              transform: 'translateY(-1px)',
                            },
                            '&:active': { transform: 'translateY(0)' },
                            '&:disabled': { borderColor: alpha(theme.palette.success.main, 0.3), color: alpha(theme.palette.success.main, 0.5) },
                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        >
                          Restore Selected
                        </Button>
                        <Button
                          variant="contained"
                          size="medium"
                          startIcon={<DeleteIcon />}
                          onClick={() => handleBulkDelete()}
                          disabled={selectedFiles.size === 0 || bulkActionLoading}
                          sx={{
                            bgcolor: theme.palette.error.main,
                            color: '#fff',
                            fontWeight: 600,
                            borderRadius: 3,
                            px: 3,
                            py: 1,
                            textTransform: 'none',
                            boxShadow: `0 4px 16px ${alpha(theme.palette.error.main, 0.3)}`,
                            border: 'none',
                            '&:hover': { bgcolor: theme.palette.error.dark, boxShadow: `0 6px 20px ${alpha(theme.palette.error.main, 0.4)}`, transform: 'translateY(-1px)' },
                            '&:active': { transform: 'translateY(0)' },
                            '&:disabled': { bgcolor: alpha(theme.palette.error.main, 0.3), color: alpha('#fff', 0.5), boxShadow: 'none' },
                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        >
                          Delete
                        </Button>
                      </>
                    ) : (
                      // Created/Failed/Skipped Tabs - Both Reprocess and Delete Actions
                      <>
                        <Button
                          variant="contained"
                          size="medium"
                          startIcon={<PlayArrowIcon />}
                          onClick={() => handleBulkReprocess()}
                          disabled={selectedFiles.size === 0 || bulkActionLoading}
                          sx={{
                            bgcolor: theme.palette.primary.main,
                            color: '#fff',
                            fontWeight: 600,
                            borderRadius: 3,
                            px: 3,
                            py: 1,
                            textTransform: 'none',
                            boxShadow: `0 4px 16px ${alpha(theme.palette.primary.main, 0.3)}`,
                            border: 'none',
                            '&:hover': {
                              bgcolor: theme.palette.primary.dark,
                              boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                              transform: 'translateY(-1px)',
                            },
                            '&:active': {
                              transform: 'translateY(0)',
                            },
                            '&:disabled': {
                              bgcolor: alpha(theme.palette.primary.main, 0.3),
                              color: alpha('#fff', 0.5),
                              boxShadow: 'none',
                            },
                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        >
                          Reprocess Selected
                        </Button>
                        <Button
                          variant="outlined"
                          size="medium"
                          startIcon={<DeleteIcon />}
                          onClick={() => handleBulkDelete()}
                          disabled={selectedFiles.size === 0 || bulkActionLoading}
                          sx={{
                            borderColor: theme.palette.error.main,
                            color: theme.palette.error.main,
                            fontWeight: 600,
                            borderRadius: 3,
                            px: 3,
                            py: 1,
                            textTransform: 'none',
                            '&:hover': {
                              bgcolor: alpha(theme.palette.error.main, 0.1),
                              borderColor: theme.palette.error.dark,
                              transform: 'translateY(-1px)',
                            },
                            '&:active': {
                              transform: 'translateY(0)',
                            },
                            '&:disabled': {
                              borderColor: alpha(theme.palette.error.main, 0.3),
                              color: alpha(theme.palette.error.main, 0.5),
                            },
                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        >
                          Delete Selected
                        </Button>
                      </>
                    )}
                  </Box>
                </Box>
              </Box>
            </Collapse>
          )}

          {/* Search Input for each tab */}
          <Box sx={{ mb: { xs: 2, sm: 3 }, px: { xs: 0, sm: 0 } }}>
            {tabValue === 0 ? (
              // Source Files Search with Auto Mode Toggle
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <TextField
                  fullWidth
                  size={isMobile ? "medium" : "small"}
                  placeholder={isMobile ? "🔍 Search files..." : "🔍 Search source files by name, path, or type..."}
                  value={sourceSearchQuery}
                  onChange={(e) => setSourceSearchQuery(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        {isSourceSearching ? (
                          <CircularProgress size={18} sx={{ color: 'primary.main' }} />
                        ) : (
                          <SearchIcon sx={{
                            color: sourceSearchQuery ? 'primary.main' : 'text.secondary',
                            fontSize: { xs: 18, sm: 20 },
                            transition: 'color 0.2s ease'
                          }} />
                        )}
                      </InputAdornment>
                    ),
                    sx: {
                      fontSize: { xs: '0.9rem', sm: '0.875rem' },
                      height: { xs: 48, sm: 40 },
                    }
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: { xs: 2, sm: 3 },
                      bgcolor: alpha(theme.palette.background.paper, 0.8),
                      backdropFilter: 'blur(10px)',
                      border: '1px solid',
                      borderColor: sourceSearchQuery ? 'primary.main' : alpha(theme.palette.divider, 0.3),
                      transition: 'all 0.3s ease',
                      minHeight: { xs: 48, sm: 40 },
                      '&:hover': {
                        borderColor: 'primary.main',
                        bgcolor: 'background.paper',
                        transform: isMobile ? 'none' : 'translateY(-1px)',
                        boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.15)}`,
                      },
                      '&.Mui-focused': {
                        borderColor: 'primary.main',
                        borderWidth: 2,
                        bgcolor: 'background.paper',
                        boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.2)}`,
                        transform: 'none',
                      },
                      '& .MuiOutlinedInput-notchedOutline': {
                        border: 'none',
                      },
                      '& .MuiInputBase-input': {
                        padding: { xs: '12px 14px', sm: '8.5px 14px' },
                        fontSize: { xs: '1rem', sm: '0.875rem' },
                        '&::placeholder': {
                          fontSize: { xs: '0.9rem', sm: '0.875rem' },
                          opacity: 0.7,
                        },
                      },
                    },
                  }}
                />

                {/* Auto Mode Toggle - Inline with search */}
                <Box sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 2,
                  py: 1,
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: bulkProcessing
                    ? alpha(theme.palette.primary.main, 0.3)
                    : (autoModeEnabled
                      ? alpha(theme.palette.success.main, 0.3)
                      : alpha(theme.palette.divider, 0.3)),
                  bgcolor: bulkProcessing
                    ? alpha(theme.palette.primary.main, 0.05)
                    : (autoModeEnabled
                      ? alpha(theme.palette.success.main, 0.05)
                      : 'background.paper'),
                  transition: 'all 0.2s ease',
                  minHeight: { xs: 48, sm: 40 },
                  flexShrink: 0,
                  '&:hover': {
                    borderColor: bulkProcessing
                      ? alpha(theme.palette.primary.main, 0.5)
                      : (autoModeEnabled
                        ? alpha(theme.palette.success.main, 0.5)
                        : alpha(theme.palette.primary.main, 0.3)),
                  }
                }}>
                  {bulkProcessing ? (
                    <CircularProgress
                      size={18}
                      sx={{ color: 'primary.main' }}
                    />
                  ) : (
                    <AutoModeIcon sx={{
                      fontSize: 18,
                      color: autoModeEnabled ? 'success.main' : 'text.secondary',
                      transition: 'color 0.2s ease'
                    }} />
                  )}

                  <Typography variant="body2" sx={{
                    fontWeight: 500,
                    color: bulkProcessing ? 'primary.main' : (autoModeEnabled ? 'success.main' : 'text.secondary'),
                    fontSize: '0.875rem',
                    transition: 'color 0.2s ease',
                    whiteSpace: 'nowrap'
                  }}>
                    {bulkProcessing ? 'Processing...' : 'Auto Mode'}
                  </Typography>

                  <Switch
                    checked={autoModeEnabled}
                    disabled={bulkProcessing}
                    onChange={async (e) => {
                      const newValue = e.target.checked;
                      setAutoModeEnabled(newValue);
                      await updateAutoModeSetting(newValue);

                      if (newValue) {
                        bulkAutoProcessFiles();
                      }
                    }}
                    color="success"
                    size="small"
                    sx={{
                      '& .MuiSwitch-switchBase.Mui-checked': {
                        color: 'success.main',
                      },
                      '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                        backgroundColor: 'success.main',
                      },
                    }}
                  />
                </Box>


              </Box>
            ) : (
              // File Operations Search
              (<TextField
                fullWidth
                size={isMobile ? "medium" : "small"}
                placeholder={isMobile
                  ? `🔍 Search ${['', 'created', 'failed', 'skipped', 'deleted'][tabValue]}...`
                  : `🔍 Search ${['', 'created', 'failed', 'skipped', 'deleted'][tabValue]} operations by filename, path, or error message...`
                }
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      {isSearching ? (
                        <CircularProgress size={18} sx={{ color: 'primary.main' }} />
                      ) : (
                        <SearchIcon sx={{
                          color: searchQuery ? 'primary.main' : 'text.secondary',
                          fontSize: { xs: 18, sm: 20 },
                          transition: 'color 0.2s ease'
                        }} />
                      )}
                    </InputAdornment>
                  ),
                  sx: {
                    fontSize: { xs: '0.9rem', sm: '0.875rem' },
                    height: { xs: 48, sm: 40 },
                  }
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: { xs: 2, sm: 3 },
                    bgcolor: alpha(theme.palette.background.paper, 0.8),
                    backdropFilter: 'blur(10px)',
                    border: '1px solid',
                    borderColor: searchQuery ? 'primary.main' : alpha(theme.palette.divider, 0.3),
                    transition: 'all 0.3s ease',
                    minHeight: { xs: 48, sm: 40 },
                    '&:hover': {
                      borderColor: 'primary.main',
                      bgcolor: 'background.paper',
                      transform: isMobile ? 'none' : 'translateY(-1px)',
                      boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.15)}`,
                    },
                    '&.Mui-focused': {
                      borderColor: 'primary.main',
                      borderWidth: 2,
                      bgcolor: 'background.paper',
                      boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.2)}`,
                      transform: 'none',
                    },
                    '& .MuiOutlinedInput-notchedOutline': {
                      border: 'none',
                    },
                    '& .MuiInputBase-input': {
                      padding: { xs: '12px 14px', sm: '8.5px 14px' },
                      fontSize: { xs: '1rem', sm: '0.875rem' },
                      '&::placeholder': {
                        fontSize: { xs: '0.9rem', sm: '0.875rem' },
                        opacity: 0.7,
                      },
                    },
                  },
                }}
              />)
            )}

            {/* Search results count for mobile */}
            {isMobile && (searchQuery || sourceSearchQuery) && (
              <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center' }}>
                <Chip
                  size="small"
                  label={
                    tabValue === 0
                      ? `${filteredSourceFiles.length} files found`
                      : `${operations.length} operations found`
                  }
                  sx={{
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    color: 'primary.main',
                    fontSize: '0.75rem',
                    height: 24,
                  }}
                />
              </Box>
            )}
          </Box>

          {/* Tab Panels */}
          <TabPanel value={tabValue} index={0}>
            {/* Modern Source File Browser with Card Layout */}
            <Box sx={{ px: { xs: 0, sm: 1, md: 2 } }}>

              {/* Loading state */}
              {sourceLoading && (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
                  <CircularProgress />
                </Box>
              )}

              {/* Error state */}
              {sourceError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {sourceError}
                </Alert>
              )}

              {/* File Cards */}
              {!sourceLoading && !sourceError && (
                <>
                  {sourceFiles.length === 0 ? (
                    <Box
                      sx={{
                        textAlign: 'center',
                        py: { xs: 6, sm: 8 },
                        px: { xs: 3, sm: 4 },
                        bgcolor: hasSourceDirectories !== false
                          ? `linear-gradient(135deg, ${alpha(theme.palette.success.main, 0.05)} 0%, ${alpha(theme.palette.success.light, 0.02)} 100%)`
                          : 'background.paper',
                        borderRadius: 3,
                        border: '1px solid',
                        borderColor: hasSourceDirectories !== false
                          ? alpha(theme.palette.success.main, 0.2)
                          : 'divider',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': hasSourceDirectories !== false ? {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          background: `radial-gradient(circle at 30% 20%, ${alpha(theme.palette.success.main, 0.1)} 0%, transparent 50%)`,
                          pointerEvents: 'none',
                        } : {},
                      }}
                    >
                      {/* Icon based on state */}
                      <Box sx={{ mb: 3 }}>
                        {hasSourceDirectories === false ? (
                          <Box sx={{
                            p: 2,
                            borderRadius: '50%',
                            bgcolor: alpha(theme.palette.warning.main, 0.1),
                            border: `2px solid ${alpha(theme.palette.warning.main, 0.3)}`,
                            display: 'inline-flex'
                          }}>
                            <SettingsIcon sx={{ fontSize: 48, color: 'warning.main' }} />
                          </Box>
                        ) : (
                          <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                            <Box sx={{
                              p: 2,
                              borderRadius: '50%',
                              bgcolor: alpha(theme.palette.success.main, 0.15),
                              border: `3px solid ${alpha(theme.palette.success.main, 0.4)}`,
                              display: 'inline-flex',
                              animation: 'celebrate 3s ease-in-out infinite',
                              '@keyframes celebrate': {
                                '0%, 100%': { transform: 'scale(1) rotate(0deg)' },
                                '25%': { transform: 'scale(1.05) rotate(2deg)' },
                                '75%': { transform: 'scale(1.05) rotate(-2deg)' },
                              },
                            }}>
                              <LinkIcon sx={{ fontSize: 48, color: 'success.main' }} />
                            </Box>
                            {/* Floating particles effect */}
                            <Box
                              sx={{
                                position: 'absolute',
                                top: -10,
                                right: -5,
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: 'success.light',
                                animation: 'float1 2s ease-in-out infinite',
                                '@keyframes float1': {
                                  '0%, 100%': { transform: 'translateY(0px) scale(1)', opacity: 0.7 },
                                  '50%': { transform: 'translateY(-10px) scale(1.2)', opacity: 1 },
                                },
                              }}
                            />
                            <Box
                              sx={{
                                position: 'absolute',
                                bottom: -5,
                                left: -10,
                                width: 6,
                                height: 6,
                                borderRadius: '50%',
                                bgcolor: 'success.main',
                                animation: 'float2 2.5s ease-in-out infinite',
                                '@keyframes float2': {
                                  '0%, 100%': { transform: 'translateY(0px) scale(1)', opacity: 0.5 },
                                  '50%': { transform: 'translateY(-15px) scale(1.3)', opacity: 1 },
                                },
                              }}
                            />
                          </Box>
                        )}
                      </Box>

                      <Typography variant="h5" sx={{
                        mb: 2,
                        fontSize: { xs: '1.25rem', sm: '1.5rem' },
                        fontWeight: 600,
                        color: hasSourceDirectories === false ? 'text.secondary' : 'success.main'
                      }}>
                        {hasSourceDirectories === false
                          ? '🔧 No Source Directories Configured'
                          : '🎉 All Source Files Tracked & Symlinked!'}
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{
                        fontSize: { xs: '0.9rem', sm: '1rem' },
                        lineHeight: 1.6,
                        maxWidth: 500,
                        mx: 'auto',
                        mb: 1
                      }}>
                        {hasSourceDirectories === false
                          ? 'Please configure SOURCE_DIR in your environment settings to start organizing your media files.'
                          : 'Perfect! All media files in your source directories have been successfully tracked and symlinked to your organized media library.'}
                      </Typography>
                      {hasSourceDirectories !== false && (
                        <Typography variant="body2" color="text.secondary" sx={{
                          fontSize: { xs: '0.8rem', sm: '0.875rem' },
                          lineHeight: 1.5,
                          maxWidth: 450,
                          mx: 'auto',
                          fontStyle: 'italic',
                          opacity: 0.8
                        }}>
                          Your original files remain in the source directories, while organized symlinks are available in your media library.
                        </Typography>
                      )}
                      {hasSourceDirectories !== false && (
                        <Box sx={{ mt: 3 }}>
                          <Box sx={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 1,
                            px: 3,
                            py: 1.5,
                            borderRadius: 3,
                            bgcolor: alpha(theme.palette.success.main, 0.1),
                            border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
                          }}>
                            <LinkIcon sx={{ fontSize: 20, color: 'success.main' }} />
                            <Typography variant="body1" color="success.main" sx={{
                              fontSize: { xs: '0.9rem', sm: '1rem' },
                              fontWeight: 600
                            }}>
                              {sourceTotalFiles > 0
                                ? `${sourceTotalFiles.toLocaleString()} files tracked & symlinked`
                                : 'All source files tracked & symlinked'}
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary" sx={{
                            mt: 2,
                            fontSize: { xs: '0.8rem', sm: '0.875rem' },
                            fontStyle: 'italic'
                          }}>
                            Your media library is fully organized and up to date!
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  ) : (
                    <>
                      {/* Processing animations */}
                      <AnimatePresence>
                        {Array.from(processingFiles.entries()).map(([filePath, fileData]) => (
                          <Box key={`processing-${filePath}`} sx={{ mb: 1.5 }}>
                            <ProcessingAnimation
                              fileName={fileData.fileName}
                              mediaName={fileData.mediaName}
                              mediaType={fileData.mediaType}
                              onComplete={() => {
                                setProcessingFiles(prev => {
                                  const newMap = new Map(prev);
                                  newMap.delete(filePath);
                                  return newMap;
                                });
                              }}
                              duration={3000}
                            />
                          </Box>
                        ))}
                      </AnimatePresence>

                      {/* Source files list */}
                      <AnimatePresence>
                        {filteredSourceFiles.map((file, _index) => (
                          <Card
                            key={file.name}
                            sx={{
                              mb: 1.5,
                              borderRadius: 3,
                              border: '1px solid',
                              borderColor: getProcessingStatus(file)?.status === 'processed' || getProcessingStatus(file)?.status === 'created'
                                ? alpha(theme.palette.success.main, 0.3)
                                : 'divider',
                              bgcolor: 'background.paper',
                              overflow: 'hidden',
                              cursor: (file.type === 'directory' || file.isSourceRoot || (file.isMediaFile && autoModeEnabled)) ? 'pointer' : 'default',
                              '&:hover': {
                                borderColor: getProcessingStatus(file)?.status === 'processed' || getProcessingStatus(file)?.status === 'created'
                                  ? alpha(theme.palette.success.main, 0.5)
                                  : 'primary.main',
                                transform: 'translateY(-2px)',
                                boxShadow: getProcessingStatus(file)?.status === 'processed' || getProcessingStatus(file)?.status === 'created'
                                  ? `0 4px 20px ${alpha(theme.palette.success.main, 0.15)}`
                                  : '0 8px 25px rgba(0, 0, 0, 0.15)',
                              },
                              transition: 'all 0.3s ease',
                            }}
                            onClick={() => handleSourceFileClick(file)}
                          >
                            <CardContent sx={{ p: { xs: 1.5, sm: 2 }, '&:last-child': { pb: { xs: 1.5, sm: 2 } } }}>
                              <Box sx={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                              }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: { xs: 1, sm: 1.5 }, minWidth: 0, flex: 1 }}>
                                  {/* Checkbox for bulk selection - only show for media files */}
                                  {file.isMediaFile && file.fullPath && (
                                    <Checkbox
                                      checked={selectedSourceFiles.has(file.fullPath)}
                                      onChange={(e) => {
                                        e.stopPropagation();
                                        handleSourceFileSelect(file.fullPath!, e.target.checked);
                                      }}
                                      size="small"
                                      sx={{
                                        p: 0.5,
                                        color: 'text.secondary',
                                        '&.Mui-checked': {
                                          color: 'primary.main',
                                        },
                                      }}
                                    />
                                  )}
                                  <Box sx={{
                                    p: { xs: 0.75, sm: 1 },
                                    borderRadius: 2,
                                    bgcolor: alpha(
                                      file.isSourceRoot ? theme.palette.primary.main :
                                        file.type === 'directory' ? theme.palette.warning.main :
                                          file.isMediaFile ? (getProcessingStatus(file)?.status === 'processed' || getProcessingStatus(file)?.status === 'created'
                                            ? theme.palette.success.main
                                            : theme.palette.info.main) :
                                            theme.palette.grey[500], 0.1
                                    ),
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                  }}>
                                    {getFileIcon(file)}
                                  </Box>
                                  <Box sx={{ minWidth: 0, flex: 1 }}>
                                    <Typography variant="body2" fontWeight="600" sx={{
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap',
                                      color: file.isSourceRoot ? 'primary.main' : 'text.primary',
                                      mb: 0.5,
                                      fontSize: { xs: '0.875rem', sm: '1rem' },
                                    }}>
                                      {file.name}
                                    </Typography>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: { xs: 1, sm: 1.5 }, flexWrap: 'wrap' }}>
                                      {getFileTypeChip(file)}
                                      {getProcessingStatusChip(file) && (
                                        <Tooltip title={getStatusTooltip(file)} arrow placement="top">
                                          <Box>{getProcessingStatusChip(file)}</Box>
                                        </Tooltip>
                                      )}
                                      {!getProcessingStatusChip(file) && file.isMediaFile && (
                                        <Tooltip title="This file has not been processed yet" arrow placement="top">
                                          <Box>{getProcessingStatusChip(file)}</Box>
                                        </Tooltip>
                                      )}
                                      {file.type === 'directory' && !file.isSourceRoot && (
                                        <Chip
                                          label="Folder"
                                          size="small"
                                          color="warning"
                                          variant="outlined"
                                          icon={<FolderIcon />}
                                          sx={{
                                            borderRadius: 2,
                                            fontWeight: 500,
                                            fontSize: { xs: '0.7rem', sm: '0.75rem' },
                                          }}
                                        />
                                      )}
                                    </Box>

                                    {/* File details */}
                                    <Stack direction="row" spacing={{ xs: 1, sm: 2 }} sx={{ mt: { xs: 0.5, sm: 1 }, flexWrap: 'wrap', gap: { xs: 0.5, sm: 1 } }}>
                                      {file.fullPath && (
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, maxWidth: '100%' }}>
                                          <InfoIcon sx={{ fontSize: { xs: 10, sm: 12 }, color: 'text.secondary', flexShrink: 0 }} />
                                          <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{
                                              fontSize: { xs: '0.7rem', sm: '0.75rem' },
                                              fontFamily: 'monospace',
                                              overflow: 'hidden',
                                              textOverflow: 'ellipsis',
                                              whiteSpace: 'nowrap',
                                              maxWidth: '100%'
                                            }}
                                            title={file.fullPath}
                                          >
                                            {file.fullPath}
                                          </Typography>
                                        </Box>
                                      )}
                                      {file.size && (
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                          <DatabaseIcon sx={{ fontSize: { xs: 10, sm: 12 }, color: 'text.secondary' }} />
                                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500, fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
                                            {file.size}
                                          </Typography>
                                        </Box>
                                      )}
                                      {file.modified && (
                                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
                                          Modified: {new Date(file.modified).toLocaleDateString()}
                                        </Typography>
                                      )}
                                      {file.isMediaFile && (
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                          <PlayCircleIcon sx={{ fontSize: { xs: 10, sm: 12 }, color: 'success.main' }} />
                                          <Typography variant="caption" color="success.main" sx={{ fontWeight: 500, fontSize: { xs: '0.7rem', sm: '0.75rem' } }}>
                                            Original Media File
                                          </Typography>
                                        </Box>
                                      )}
                                    </Stack>
                                  </Box>
                                </Box>

                                {/* Action buttons for media files */}
                                {file.isMediaFile && (
                                  <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                                    {/* Show auto-processing indicator */}
                                    {autoProcessingFiles.has(file.fullPath!) ? (
                                      <Tooltip title="Auto-processing...">
                                        <Box sx={{
                                          display: 'flex',
                                          alignItems: 'center',
                                          gap: 1,
                                          px: 1.5,
                                          py: 0.5,
                                          borderRadius: 2,
                                          bgcolor: alpha(theme.palette.success.main, 0.1),
                                          border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`
                                        }}>
                                          <CircularProgress
                                            size={16}
                                            sx={{ color: 'success.main' }}
                                          />
                                          <Typography variant="caption" color="success.main" sx={{ fontWeight: 600 }}>
                                            Auto
                                          </Typography>
                                        </Box>
                                      </Tooltip>
                                    ) : (
                                      <Tooltip title={autoModeEnabled ? "Click to auto-process" : "Process File"}>
                                        <IconButton
                                          size="small"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            if (autoModeEnabled) {
                                              autoProcessFile(file);
                                            } else {
                                              handleProcessFile(file);
                                            }
                                          }}
                                          sx={{
                                            bgcolor: alpha(
                                              autoModeEnabled ? theme.palette.success.main : theme.palette.primary.main,
                                              0.1
                                            ),
                                            color: autoModeEnabled ? 'success.main' : 'primary.main',
                                            '&:hover': {
                                              bgcolor: alpha(
                                                autoModeEnabled ? theme.palette.success.main : theme.palette.primary.main,
                                                0.2
                                              ),
                                              transform: 'scale(1.1)',
                                            },
                                            transition: 'all 0.2s ease'
                                          }}
                                        >
                                          {autoModeEnabled ? (
                                            <AutoModeIcon sx={{ fontSize: 18 }} />
                                          ) : (
                                            <SettingsIcon sx={{ fontSize: 18 }} />
                                          )}
                                        </IconButton>
                                      </Tooltip>
                                    )}
                                  </Box>
                                )}
                              </Box>
                            </CardContent>
                          </Card>
                        ))}
                      </AnimatePresence>
                    </>
                  )}
                </>
              )}

              {/* Pagination */}
              {sourceTotalPages > 1 && (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                  <Pagination
                    count={sourceTotalPages}
                    page={sourcePage}
                    onChange={(_, newPage) => setSourcePage(newPage)}
                    color="primary"
                    size={isMobile ? "small" : "medium"}
                    siblingCount={isMobile ? 0 : 1}
                    boundaryCount={isMobile ? 1 : 1}
                    sx={{
                      '& .MuiPaginationItem-root': {
                        borderRadius: 2,
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        minWidth: { xs: 28, sm: 32 },
                        height: { xs: 28, sm: 32 },
                      },
                      '& .MuiPagination-ul': {
                        gap: { xs: 0.25, sm: 0.5 }
                      }
                    }}
                  />
                </Box>
              )}

              {/* Summary */}
              {filteredSourceFiles.length > 0 && (
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 3 }}>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      whiteSpace: { xs: 'normal', sm: 'nowrap' },
                      textAlign: 'center',
                      fontSize: { xs: '0.75rem', sm: '0.875rem' }
                    }}
                  >
                    Showing {filteredSourceFiles.length} of {sourceTotalFiles.toLocaleString()} items
                    {sourceSearchQuery && ` (filtered)`}
                  </Typography>
                </Box>
              )}
            </Box>
          </TabPanel>

          {tabValue === 1 && (
            <TabPanel value={tabValue} index={1}>
              {renderFileTable(operations, searchQuery ? 'No created files match your search' : 'No files created yet')}
            </TabPanel>
          )}

          {tabValue === 2 && (
            <TabPanel value={tabValue} index={2}>
              {renderFileTable(operations, searchQuery ? 'No failed operations match your search' : 'No failed file operations')}
            </TabPanel>
          )}

          {tabValue === 3 && (
            <TabPanel value={tabValue} index={3}>
              {renderFileTable(operations, searchQuery ? 'No skipped files match your search' : 'No skipped files')}
            </TabPanel>
          )}

          {tabValue === 4 && (
            <TabPanel value={tabValue} index={4}>
              {renderFileTable(operations, searchQuery ? 'No deleted files match your search' : 'No deleted files')}
            </TabPanel>
          )}

          {/* Pagination and Summary for Operations - Only show for operation tabs */}
          {tabValue > 0 && (
            <Box sx={{
              mt: 3,
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              justifyContent: 'center',
              alignItems: 'center',
              gap: { xs: 2, sm: 3 }
            }}>
              {Math.ceil(totalOperations / recordsPerPage) > 1 && (
                <Pagination
                  count={Math.ceil(totalOperations / recordsPerPage)}
                  page={currentPage}
                  onChange={(_, page) => setCurrentPage(page)}
                  color="primary"
                  size={isMobile ? "small" : "medium"}
                  siblingCount={isMobile ? 0 : 1}
                  boundaryCount={isMobile ? 1 : 1}
                  sx={{
                    '& .MuiPaginationItem-root': {
                      borderRadius: 2,
                      fontSize: { xs: '0.75rem', sm: '0.875rem' },
                      minWidth: { xs: 28, sm: 32 },
                      height: { xs: 28, sm: 32 },
                    },
                    '& .MuiPagination-ul': {
                      gap: { xs: 0.25, sm: 0.5 }
                    }
                  }}
                />
              )}
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{
                  whiteSpace: { xs: 'normal', sm: 'nowrap' },
                  textAlign: 'center',
                  fontSize: { xs: '0.75rem', sm: '0.875rem' }
                }}
              >
                Showing {operations.length} of {totalOperations.toLocaleString()} operations
                {searchQuery && ` (filtered)`}
              </Typography>
            </Box>
          )}

          {/* Mobile Floating Action Buttons for Operations */}
          {isMobile && (
            <>
              {/* Bulk Delete FAB for Skipped Files on Mobile */}
              {mainTabValue === 0 && tabValue === 3 && statusCounts.skipped > 0 && (
                <MotionFab
                  color="error"
                  aria-label="delete all skipped"
                  onClick={() => setBulkDeleteDialogOpen(true)}
                  disabled={bulkDeleteLoading}
                  sx={{
                    position: 'fixed',
                    bottom: 24,
                    right: 88,
                    zIndex: 1000,
                    background: 'linear-gradient(45deg, #ef4444 30%, #dc2626 90%)',
                    boxShadow: '0 8px 16px 0 rgba(239, 68, 68, 0.3)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #dc2626 30%, #b91c1c 90%)',
                      boxShadow: '0 12px 20px 0 rgba(239, 68, 68, 0.4)',
                    },
                    '&:disabled': {
                      background: 'rgba(239, 68, 68, 0.3)',
                    },
                  }}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {bulkDeleteLoading ? (
                    <CircularProgress size={24} sx={{ color: 'white' }} />
                  ) : (
                    <DeleteSweepIcon sx={{ fontSize: 24 }} />
                  )}
                </MotionFab>
              )}

              {/* Manual Import FAB */}
              <MotionFab
                color="secondary"
                aria-label="manual import"
                onClick={() => setManualImportOpen(true)}
                sx={{
                  position: 'fixed',
                  bottom: 24,
                  right: 88,
                  zIndex: 1000,
                  background: 'linear-gradient(45deg, #10B981 30%, #059669 90%)',
                  boxShadow: '0 8px 16px 0 rgba(16, 185, 129, 0.3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #059669 30%, #047857 90%)',
                    boxShadow: '0 12px 20px 0 rgba(16, 185, 129, 0.4)',
                  },
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <AssignmentIcon sx={{ fontSize: 24 }} />
              </MotionFab>

              {/* Refresh FAB */}
              <MotionFab
                color="primary"
                aria-label="refresh"
                onClick={() => fetchFileOperations(true)}
                disabled={loading}
                sx={{
                  position: 'fixed',
                  bottom: 24,
                  right: 24,
                  zIndex: 1000,
                  background: 'linear-gradient(45deg, #6366F1 30%, #8B5CF6 90%)',
                  boxShadow: '0 8px 16px 0 rgba(99, 102, 241, 0.3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #5B5FE8 30%, #7C3AED 90%)',
                    boxShadow: '0 12px 20px 0 rgba(99, 102, 241, 0.4)',
                  },
                  '&:disabled': {
                    background: 'rgba(99, 102, 241, 0.3)',
                  },
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <RefreshIcon sx={{
                  fontSize: 24,
                  animation: loading ? 'spin 1s linear infinite' : 'none',
                  '@keyframes spin': {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' },
                  },
                }} />
              </MotionFab>
            </>
          )}

          {/* Desktop Floating Action Buttons */}
          {!isMobile && (
            <>
              {/* Manual Import FAB */}
              <MotionFab
                color="secondary"
                aria-label="manual import"
                onClick={() => setManualImportOpen(true)}
                sx={{
                  position: 'fixed',
                  bottom: 24,
                  right: 88,
                  zIndex: 1000,
                  background: 'linear-gradient(45deg, #10B981 30%, #059669 90%)',
                  boxShadow: '0 8px 16px 0 rgba(16, 185, 129, 0.3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #059669 30%, #047857 90%)',
                    boxShadow: '0 12px 20px 0 rgba(16, 185, 129, 0.4)',
                  },
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <AssignmentIcon sx={{ fontSize: 24 }} />
              </MotionFab>

              {/* Refresh FAB */}
              <MotionFab
                color="primary"
                aria-label="refresh"
                onClick={() => fetchFileOperations(true)}
                disabled={loading}
                sx={{
                  position: 'fixed',
                  bottom: 24,
                  right: 24,
                  zIndex: 1000,
                  background: 'linear-gradient(45deg, #6366F1 30%, #8B5CF6 90%)',
                  boxShadow: '0 8px 16px 0 rgba(99, 102, 241, 0.3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #5B5FE8 30%, #7C3AED 90%)',
                    boxShadow: '0 12px 20px 0 rgba(99, 102, 241, 0.4)',
                  },
                  '&:disabled': {
                    background: 'rgba(99, 102, 241, 0.3)',
                  },
                }}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <RefreshIcon sx={{
                  fontSize: 24,
                  animation: loading ? 'spin 1s linear infinite' : 'none',
                  '@keyframes spin': {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' },
                  },
                }} />
              </MotionFab>
            </>
          )}
        </>
      )}
      {/* Database Tab Content */}
      {mainTabValue === 1 && (
        <DatabaseSearch />
      )}


      {/* ModifyDialog for file processing */}
      {modifyDialogOpen && (
        <ModifyDialog
          open={modifyDialogOpen}
          onClose={handleModifyDialogClose}
          currentFilePath={currentFileForProcessing}
          useManualSearch={tabValue === 2}
          onNavigateBack={() => {
          }}
        />
      )}

      {/* ModifyDialog for bulk operations */}
      {bulkModifyDialogOpen && bulkModifyFilePaths.length > 0 && (
        <ModifyDialog
          open={bulkModifyDialogOpen}
          onClose={handleBulkModifyDialogClose}
          currentFilePath={bulkModifyFilePaths[0]}
          bulkFilePaths={bulkModifyFilePaths}
          useBatchApply={true}
          useManualSearch={false}
          onNavigateBack={() => { }}
        />
      )}


      {/* Bulk Delete Confirmation Dialog */}
      <Dialog
        open={bulkDeleteDialogOpen}
        onClose={() => setBulkDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 3,
            backgroundColor: theme.palette.background.paper,
            background: theme.palette.mode === 'dark'
              ? `linear-gradient(135deg, ${theme.palette.background.paper} 0%, rgba(239, 68, 68, 0.08) 100%)`
              : `linear-gradient(135deg, #ffffff 0%, #fef2f2 100%)`,
            border: theme.palette.mode === 'dark'
              ? `2px solid ${theme.palette.error.main}40`
              : `2px solid #fecaca`,
            boxShadow: theme.palette.mode === 'dark'
              ? '0 8px 32px rgba(0, 0, 0, 0.6)'
              : 'none',
            backdropFilter: 'blur(10px)',
          }
        }}
        slotProps={{
          backdrop: {
            sx: {
              backgroundColor: theme.palette.mode === 'dark'
                ? 'rgba(0, 0, 0, 0.7)'
                : 'rgba(0, 0, 0, 0.5)',
              backdropFilter: 'blur(4px)',
            }
          }
        }}
      >
        <DialogTitle sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          color: 'error.main',
          fontWeight: 600,
          backgroundColor: theme.palette.mode === 'dark'
            ? 'rgba(239, 68, 68, 0.15)'
            : '#fef2f2',
          borderBottom: theme.palette.mode === 'dark'
            ? `2px solid ${theme.palette.error.main}60`
            : `2px solid #fecaca`,
        }}>
          <DeleteSweepIcon />
          Delete All Skipped Files
        </DialogTitle>
        <DialogContent sx={{
          background: 'transparent',
        }}>
          <DialogContentText sx={{ mb: 2, color: 'text.primary' }}>
            Are you sure you want to delete all <strong>{statusCounts.skipped}</strong> skipped files from the database?
          </DialogContentText>
          <DialogContentText sx={{
            color: 'text.primary',
            fontSize: '0.875rem',
            bgcolor: theme.palette.mode === 'dark'
              ? alpha(theme.palette.background.default, 0.5)
              : '#f9fafb',
            p: 2,
            borderRadius: 2,
            borderLeft: theme.palette.mode === 'dark'
              ? `4px solid ${theme.palette.error.main}`
              : `4px solid #f87171`,
            backdropFilter: 'blur(10px)',
          }}>
            This action will permanently remove all skipped file records from the database.
            The original files will remain untouched in their source locations.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{
          p: 3,
          pt: 1,
          gap: 1,
          background: theme.palette.mode === 'dark'
            ? alpha(theme.palette.background.default, 0.3)
            : '#f9fafb',
          backdropFilter: 'blur(10px)',
        }}>
          <Button
            onClick={() => setBulkDeleteDialogOpen(false)}
            disabled={bulkDeleteLoading}
            sx={{ textTransform: 'none' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleBulkDeleteSkippedFiles}
            color="error"
            variant="contained"
            disabled={bulkDeleteLoading}
            startIcon={bulkDeleteLoading ? <CircularProgress size={16} /> : <DeleteSweepIcon />}
            sx={{
              textTransform: 'none',
              fontWeight: 600,
              borderRadius: 2,
              ...(theme.palette.mode === 'light' && {
                bgcolor: '#ef4444',
                color: '#ffffff',
                border: 'none',
                boxShadow: 'none',
                '&:hover': {
                  bgcolor: '#dc2626',
                  border: 'none',
                  boxShadow: 'none',
                },
                '&:disabled': {
                  bgcolor: alpha('#ef4444', 0.3),
                  color: alpha('#ffffff', 0.7),
                  border: 'none',
                  boxShadow: 'none',
                },
              })
            }}
          >
            {bulkDeleteLoading ? 'Deleting...' : 'Delete All'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Bulk Action Confirmation Dialog */}
      <Dialog
        open={bulkDeleteDialogOpen}
        onClose={() => setBulkDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 3,
            backgroundColor: theme.palette.background.paper,
            background: theme.palette.mode === 'dark'
              ? `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${alpha(theme.palette.error.main, 0.08)} 100%)`
              : `linear-gradient(135deg, #ffffff 0%, #fef2f2 100%)`,
            border: theme.palette.mode === 'dark'
              ? `2px solid ${theme.palette.error.main}40`
              : `2px solid #fecaca`,
            boxShadow: theme.palette.mode === 'dark'
              ? '0 8px 32px rgba(0, 0, 0, 0.6)'
              : '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
            backdropFilter: 'blur(10px)',
          }
        }}
        slotProps={{
          backdrop: {
            sx: {
              backgroundColor: theme.palette.mode === 'dark'
                ? 'rgba(0, 0, 0, 0.7)'
                : 'rgba(0, 0, 0, 0.5)',
              backdropFilter: 'blur(4px)',
            }
          }
        }}
      >
        <DialogTitle sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          color: 'error.main',
          fontWeight: 600,
          backgroundColor: theme.palette.mode === 'dark'
            ? alpha(theme.palette.error.main, 0.15)
            : '#fef2f2',
          borderBottom: theme.palette.mode === 'dark'
            ? `2px solid ${theme.palette.error.main}60`
            : `2px solid #fecaca`,
        }}>
          <DeleteIcon />
          Delete Selected Files
        </DialogTitle>
        <DialogContent sx={{
          background: 'transparent',
        }}>
          <DialogContentText sx={{ mb: 2, color: 'text.primary' }}>
            Are you sure you want to delete <strong>{selectedFiles.size}</strong> selected files?
          </DialogContentText>
          <DialogContentText sx={{
            color: 'text.primary',
            fontSize: '0.875rem',
            bgcolor: theme.palette.mode === 'dark'
              ? alpha(theme.palette.background.default, 0.5)
              : '#f9fafb',
            p: 2,
            borderRadius: 2,
            borderLeft: theme.palette.mode === 'dark'
              ? `4px solid ${theme.palette.error.main}`
              : `4px solid #f87171`,
            backdropFilter: 'blur(10px)',
          }}>
            This action will permanently remove the selected file records from the database. The original files will remain untouched in their source locations.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{
          p: 3,
          pt: 1,
          gap: 1,
          background: theme.palette.mode === 'dark'
            ? alpha(theme.palette.background.default, 0.3)
            : '#f9fafb',
          backdropFilter: 'blur(10px)',
        }}>
          <Button
            onClick={() => setBulkDeleteDialogOpen(false)}
            disabled={bulkActionLoading}
            sx={{ textTransform: 'none' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleBulkDeleteConfirm}
            color="error"
            variant="contained"
            disabled={bulkActionLoading}
            startIcon={bulkActionLoading ? <CircularProgress size={16} /> : <DeleteIcon />}
            sx={{
              textTransform: 'none',
              fontWeight: 600,
              borderRadius: 2,
              ...(theme.palette.mode === 'light' && {
                bgcolor: '#ef4444',
                color: '#ffffff',
                border: 'none',
                boxShadow: 'none',
                '&:hover': {
                  bgcolor: '#dc2626',
                  border: 'none',
                  boxShadow: 'none',
                },
                '&:disabled': {
                  bgcolor: alpha('#ef4444', 0.3),
                  color: alpha('#ffffff', 0.7),
                  border: 'none',
                  boxShadow: 'none',
                },
              })
            }}
          >
            {bulkActionLoading ? 'Deleting...' : 'Delete Selected'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Success Dialog */}
      <Dialog
        open={successDialogOpen}
        onClose={() => setSuccessDialogOpen(false)}
        maxWidth="xs"
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            bgcolor: theme.palette.mode === 'dark' ? '#000000' : theme.palette.background.paper,
            backgroundColor: theme.palette.mode === 'dark' ? '#000000' : theme.palette.background.paper,
            backgroundImage: 'none',
            borderRadius: 2,
            minWidth: 400,
            boxShadow: theme.palette.mode === 'dark'
              ? '0 8px 32px rgba(0, 0, 0, 0.8)'
              : '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
          }
        }}
        PaperProps={{
          sx: {
            bgcolor: theme.palette.mode === 'dark' ? '#000000' : theme.palette.background.paper,
            backgroundColor: theme.palette.mode === 'dark' ? '#000000' : theme.palette.background.paper,
            backgroundImage: 'none',
            borderRadius: 2,
            minWidth: 400,
            boxShadow: theme.palette.mode === 'dark'
              ? '0 8px 32px rgba(0, 0, 0, 0.8)'
              : '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
          }
        }}
      >
        <DialogTitle sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          color: '#4caf50',
          fontWeight: 'bold'
        }}>
          <Avatar sx={{ bgcolor: '#4caf50', width: 32, height: 32 }}>
            <CheckCircle sx={{ color: theme.palette.mode === 'dark' ? '#ffffff' : '#000000' }} />
          </Avatar>
          Operation Complete
        </DialogTitle>
        <DialogContent sx={{ pb: 1 }}>
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <Avatar sx={{
              bgcolor: '#4caf50',
              width: 80,
              height: 80,
              margin: '0 auto 16px auto'
            }}>
              <CheckCircle sx={{ fontSize: 40, color: theme.palette.mode === 'dark' ? '#ffffff' : '#000000' }} />
            </Avatar>
            <Typography variant="h6" sx={{ color: '#4caf50', fontWeight: 'bold', mb: 1 }}>
              {successMessage}
            </Typography>
            <Typography variant="body2" sx={{ color: theme.palette.mode === 'dark' ? '#aaa' : '#666' }}>
              This dialog will close automatically in a few seconds...
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions sx={{ justifyContent: 'center', pb: 2 }}>
          <Button
            onClick={() => setSuccessDialogOpen(false)}
            variant="contained"
            sx={{
              backgroundColor: '#2196f3',
              '&:hover': { backgroundColor: '#1976d2' },
              minWidth: 100
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Manual Import Dialog */}
      <ManualImport
        open={manualImportOpen}
        onClose={() => setManualImportOpen(false)}
      />

    </Box>
  );
}

export { FileOperations };
export default FileOperations;
